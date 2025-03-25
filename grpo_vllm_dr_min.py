from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, random, time, atexit
import torch
import numpy as np
import requests
from tqdm import tqdm
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import wandb
from dataclasses import dataclass
from typing import Optional
import re
from rich.console import Console
from math_verify import parse, verify, ExprExtractionConfig

from vllm_client import VLLMClient
from utils import print_prompt_completions_sample

console = Console()

SYSTEM_PROMPT = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

@dataclass
class Args:
    """Configuration for Dr. GRPO Training with vLLM server"""
    # Model and server configuration
    model_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    vllm_host: str = "localhost"
    vllm_port: int = 8000
    device: str = "cuda"  # Use 'mps' for Mac, 'cuda' for GPU, or 'cpu'
    
    # Training hyperparameters
    learning_rate: float = 1e-6
    train_batch_size: int = 4
    num_generations: int = 8
    all_steps: int = 1000
    gen_update_steps: int = 16  # Steps between updating vLLM server weights
    
    # Generation parameters
    temperature: float = 0.9
    top_p: float = 0.9
    max_tokens: int = 700
    
    # Logging and saving
    output_dir: str = "./output"
    save_steps: int = 200
    log_steps: int = 1
    wandb_project: str = "dr-grpo"
    wandb_run_name: Optional[str] = None
    
    # Misc
    seed: int = 42


def reward_correct(item, answer):
    """Verify if the answer is mathematically correct"""
    pattern = r"\d+\.\d+|\d+/\d+|\d+"
    nums = re.findall(pattern, answer)
    if len(nums) == 0:
        return -1.0
    lastnum = nums[-1]
    try:
        ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
        ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
        return 1.0 if verify(ans, ground_truth) else -1.0
    except:
        return -1.0


def reward_format(item, answer):
    """Verify if the answer follows the required format"""
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    think_count = answer.count("<think>") + answer.count("</think>")
    answer_count = answer.count("<answer>") + answer.count("</answer>")
    return (
        1.25
        if re.match(pattern, answer, re.DOTALL | re.VERBOSE)
        and think_count == 2
        and answer_count == 2
        else -1.0
    )





def get_per_token_logps(logits, input_ids):
    """Calculate per-token log probabilities"""
    # logits shape: [batch_size, seq_len, vocab_size]
    # input_ids shape: [batch_size, seq_len]

    # Ensure input_ids has the right shape
    if input_ids.dim() == 3:
        input_ids = input_ids.squeeze(-1)  # Remove extra dimension if present

    log_probs = logits.log_softmax(dim=-1)  # [batch_size, seq_len, vocab_size]

    # Create proper indices for gathering
    indices = input_ids.unsqueeze(-1)  # [batch_size, seq_len, 1]

    # Gather the log probabilities for the actual tokens
    token_log_probs = log_probs.gather(dim=2, index=indices)  # [batch_size, seq_len, 1]
    token_log_probs = token_log_probs.squeeze(-1)  # [batch_size, seq_len]

    return token_log_probs

def prepare_batch(
    prompt_ids,
    completion_ids_list,
    rewards_list,
    tokenizer,
    device,
):
    """Prepare batch for training"""
    # Pad completions
    tensor_list = [torch.tensor(lst, device=device) for lst in completion_ids_list]
    output_ids = pad_sequence(
        tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    
    # Repeat prompts for each completion and merge
    prompt_len = prompt_ids.shape[1]
    num_generations = output_ids.shape[0] // prompt_ids.shape[0]
    question_repeated = prompt_ids.repeat(1, num_generations).view(-1, prompt_len)

    input_ids = torch.cat([question_repeated, output_ids], dim=1)


    # For logging purposes, also return the completion texts
    return {
        "prompt_len": prompt_len,
        "input_ids": input_ids,
        "rewards": torch.tensor(rewards_list, dtype=torch.float32, device=device),
    }


def Dr_GRPO_step(batch, model, tokenizer):
    """Dr. GRPO training step without normalization terms"""
    prompt_len = batch['prompt_len']
    input_ids = batch['input_ids'].to(model.device)
    advantages = batch['rewards'].to(model.device).unsqueeze(1)
    logits = model(input_ids).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_len-1:]
    completion_mask = (input_ids[:, prompt_len-1:] != tokenizer.pad_token_id).int()

    per_token_loss = - torch.exp(per_token_logps - per_token_logps.detach()) * advantages
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # Calculate mean and std of rewards
    metrics = {}
    metrics["advantage_mean"] = advantages.mean().item()
    metrics["advantage_std"] = advantages.std().item()

    # Calculate mean token length
    metrics["completion_length"] = completion_mask.sum(dim=1).float().mean().item()

    return loss, metrics




def generate_and_score_completions(
    inputs, 
    tokenizer, 
    vllm_client,
    device="cuda",
    num_generations=3,
    temperature=0.9,
    top_p=0.9,
    max_tokens=700,
):
    # Prepare prompts
    prompts = [x["Q"] for x in inputs]
    prompt_texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for x in prompts
    ]

    prompt_ids = tokenizer(
        prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False
    )["input_ids"].to(device)

    completion_ids = vllm_client.generate(
        prompt_texts,
        n=num_generations,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids]
    rewards = []
    for i, inp in enumerate(inputs):
        for comp in completions[i * num_generations : (i + 1) * num_generations]:
            rewards.append(reward_correct(inp, comp) + reward_format(inp, comp))

    return completion_ids, completions, rewards, prompts, prompt_ids


def main(args: Args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize wandb
    run_name = args.wandb_run_name or f"dr-grpo-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name, config=args.__dict__)

    # Initialize model, tokenizer and optimizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # Using float32 for Mac compatibility
        device_map=args.device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Initialize vLLM client
    vllm_client = VLLMClient(args.vllm_host, args.vllm_port)
    
    # Load dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    QAs = [
        {"Q": x, "A": y.split("####")[-1].strip()}
        for x, y in zip(dataset["question"], dataset["answer"])
    ]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    progress = tqdm(range(1, args.all_steps + 1))
    for step in progress:
        # Sample batch and generate completions
        inputs = random.sample(QAs, args.train_batch_size)
        
        # Generate and score completions
        completion_ids, completions, rewards, prompts, prompt_ids = generate_and_score_completions(
            inputs,
            tokenizer,
            vllm_client,
            device=args.device,
            num_generations=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        print_prompt_completions_sample(prompts, completions[::args.num_generations], rewards[::args.num_generations], step)

        batch = prepare_batch(
            prompt_ids,
            completion_ids,
            rewards,
            tokenizer,
            args.device,
        )

        # Training step
        loss, metrics = Dr_GRPO_step(
            batch, model, tokenizer
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.set_description(f"Loss: {loss.item():.6f}")

        # Logging
        if step % args.log_steps == 0:
            wandb.log({
                "step": step,
                **metrics,
            })

        # Update vLLM server weights
        if step % args.gen_update_steps == 0:
            print("Updating vLLM server weights...")
            state_dict = model.state_dict()
            for name, param in state_dict.items():
                vllm_client.update_named_param(name, param.data)
            vllm_client.reset_prefix_cache()

        # Save model checkpoint
        if step % args.save_steps == 0:
            print(f"Saving model at step {step}")
            save_name = f"{args.output_dir}/step_{step}"
            model.save_pretrained(save_name)
            tokenizer.save_pretrained(save_name)

    wandb.finish()


if __name__ == "__main__":
    import simple_parsing as sp
    script_args = sp.parse(Args)
    main(script_args)
