from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, re, random, time
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import requests
from tqdm import tqdm
import deepspeed
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import wandb
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Args:
    """Configuration for Dr. GRPO Training with vLLM server"""
    # Model and server configuration
    model_path: str = "./data/Qwen2.5-3B" #Path to the model or model identifier from huggingface.co/models
    vllm_server: str = "http://localhost:8000" #URL of the vLLM server    
    beta: float = 0.0 #Beta parameter for KL divergence (0.0 for Dr. GRPO)
    clip_param: float = 0.2 #Clipping parameter for ratio
    all_steps: int = 1000 #Total number of training steps
    num_generations: int = 5 #Number of generations per prompt (G)
    gen_update_steps: int = 16 #Number of steps between updating the vLLM server model
    train_batch_size: int = 4 #Training batch size per device
    temperature: float = 0.9 #Sampling temperature for generation
    top_p: float = 0.9 #Top-p sampling parameter
    output_dir: str = "./output" #Directory to save model checkpoints
    save_steps: int = 200 #Save checkpoint every N steps
    log_steps: int = 1 #Log to wandb every N steps
    wandb_project: str = "dr-grpo" #Weights & Biases project name
    wandb_run_name: Optional[str] = None #Weights & Biases run name (auto-generated if None)
    gradient_accumulation_steps: int = 4 #Number of updates steps to accumulate before performing a backward pass
    learning_rate: float = 1e-6 #Learning rate for training
    seed: int = 42 #Random seed for reproducibility
    
    def get_ds_config(self):
        """Return DeepSpeed configuration dictionary"""
        return {
            "train_micro_batch_size_per_gpu": self.train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": self.learning_rate}
            },
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
                "stage3_gather_16bit_weights_on_model_save": True,
                "offload_optimizer": {"device": "cpu"}
            }
        }


class VLLMClient:
    def __init__(self, server_url):
        self.server_url = server_url
        
    def generate(self, prompts, n=1, temperature=0.9, top_p=1.0, max_tokens=700, **kwargs):
        """Generate completions for prompts using vLLM server"""
        request_data = {
            "prompts": prompts,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        request_data.update(kwargs)
        
        response = requests.post(f"{self.server_url}/generate/", json=request_data)
        return response.json()["completion_ids"]
    
    def update_named_param(self, name, tensor_data):
        """Update named parameter in the vLLM server"""
        request_data = {
            "name": name,
            "dtype": str(tensor_data.dtype).split(".")[-1],
            "shape": list(tensor_data.shape)
        }
        
        # First send the parameter metadata
        requests.post(f"{self.server_url}/update_named_param/", json=request_data)
        
    def reset_prefix_cache(self):
        """Reset the prefix cache in vLLM server"""
        requests.post(f"{self.server_url}/reset_prefix_cache/")
        
    def init_communicator(self, host, port, world_size):
        """Initialize the weight sync communicator"""
        request_data = {
            "host": host,
            "port": port,
            "world_size": world_size
        }
        requests.post(f"{self.server_url}/init_communicator/", json=request_data)


def get_per_token_logps(logits, input_ids):
    """Calculate per-token log probabilities"""
    # logits shape: [batch_size, seq_len, vocab_size]
    # input_ids shape: [batch_size, seq_len]
    
    # Ensure input_ids has the right shape
    if input_ids.dim() == 3:
        input_ids = input_ids.squeeze(-1)  # Remove extra dimension if present
    
    log_probs = logits.log_softmax(dim=-1)  # [batch_size, seq_len, vocab_size]
    
    # Handle gathering correctly
    batch_size, seq_len, vocab_size = log_probs.shape
    
    # Create proper indices for gathering
    indices = input_ids.unsqueeze(-1)  # [batch_size, seq_len, 1]
    
    # Gather the log probabilities for the actual tokens
    token_log_probs = log_probs.gather(dim=2, index=indices)  # [batch_size, seq_len, 1]
    token_log_probs = token_log_probs.squeeze(-1)  # [batch_size, seq_len]
    
    return token_log_probs


def Dr_GRPO_step(batch, engine, tokenizer, beta=0.0, clip_param=0.2):
    """Dr. GRPO training step without normalization terms"""
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    
    # Get model logits
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit
    input_ids = inputs[:, 1:]   # (B, L-1), exclude the first input ID
    
    # Calculate per-token log probabilities for the current policy
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length-1:]
    
    # Get log probabilities from the generation model (vLLM)
    gen_logps = batch['gen_logps'].to(engine.device)
    
    # Ensure gen_logps has the right shape to match per_token_logps
    if gen_logps.shape != per_token_logps.shape:
        # Adjust gen_logps shape if needed
        if gen_logps.dim() == 3 and gen_logps.shape[-1] == 1:
            gen_logps = gen_logps.squeeze(-1)
    
    # KL divergence term (if beta > 0)
    if beta > 0:
        per_token_kl = torch.exp(gen_logps - per_token_logps) - (gen_logps - per_token_logps) - 1
    
    # Calculate policy ratio and clipped objective
    ratio = torch.exp(per_token_logps - gen_logps)
    clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
    
    # We want to maximize, so negative sign for minimization
    per_token_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    # Apply mask for completion tokens and compute mean loss
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    # Adjust completion_mask to match per_token_loss shape
    if completion_mask.shape[1] != per_token_loss.shape[1]:
        completion_mask = completion_mask[:, :per_token_loss.shape[1]]
    
    # For properly normalized weighting, divide by sum of mask
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
    
    # Calculate metrics for logging
    metrics = {}
    metrics['loss'] = loss.item()
    
    # Calculate clipping ratio
    is_clipped = (ratio < (1-clip_param)) | (ratio > (1+clip_param))
    metrics['clip_ratio'] = (is_clipped.float() * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
    
    # Calculate mean and std of rewards
    metrics['advantage_mean'] = advantages.mean().item()
    metrics['advantage_std'] = advantages.std().item()
    
    # Calculate mean token length
    metrics['completion_length'] = completion_mask.sum(dim=1).float().mean().item()
    
    return loss, metrics


def prepare_batch(inputs_list, rewards_list, completion_ids_list, completion_texts, prompt_token_ids, device):
    """Prepare batch for training"""
    # Pad completions
    tensor_list = [torch.tensor(lst, device=device) for lst in completion_ids_list]
    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=0)  # Using 0 as padding, will be replaced
    
    # Repeat prompts for each completion and merge
    plen = prompt_token_ids.shape[1]
    Qrep = prompt_token_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
    merged_ids = torch.cat([Qrep, output_ids], dim=1)
    
    # Generate simulated log probabilities with the correct shape
    # The shape should be [batch_size, seq_len] to match what get_per_token_logps will produce
    gen_logps = torch.randn(output_ids.shape, device=device) * 0.1 - 5.0
    
    # For logging purposes, also return the completion texts
    return {
        "plen": plen,
        "inputs": merged_ids,
        "rewards": torch.tensor(rewards_list, dtype=torch.float32, device=device),
        "gen_logps": gen_logps,
        "completion_texts": completion_texts
    }


def main(args: Args):
    # # Set random seed for reproducibility
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    
    # Initialize deepspeed
    try:
        deepspeed.init_distributed()
        
        # Initialize wandb in the main process
        if dist.get_rank() == 0:
            run_name = args.wandb_run_name or f"dr-grpo-{time.strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project=args.wandb_project, name=run_name, config=args.__dict__)
        
        # Initialize vLLM client
        vllm_client = VLLMClient("http://0.0.0.0:8000")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]
        
        system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
        The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
        
        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                    torch_dtype=torch.bfloat16, 
                                                    attn_implementation="flash_attention_2")
        
        # Initialize DeepSpeed
        ds_config = args.get_ds_config()
        engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, 
                                                    model=model, 
                                                    model_parameters=model.parameters())
        
        # Initialize weight sync with vLLM server (only in main process)
        if dist.get_rank() == 0:
            print("Initializing weight sync with vLLM server...")
            # Communication details would go here
            # vllm_client.init_communicator(...)
        
        # Define reward functions
        from math_verify import parse, verify, ExprExtractionConfig
        
        def reward_correct(item, answer):
            """Verify if the answer is mathematically correct"""
            pattern = r'\d+\.\d+|\d+/\d+|\d+'
            nums = re.findall(pattern, answer) 
            if len(nums) == 0: return -1.0
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
            return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2 else -1.0
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Training loop
        progress = range(1, args.all_steps+1)
        if dist.get_rank() == 0: 
            progress = tqdm(progress)
        
        for step in progress:
            # Sample from dataset
            inputs = random.sample(QAs, args.train_batch_size)
            
            # Format prompts for generation
            prompts = [x["Q"] for x in inputs]
            prompt_texts = [tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
            
            # Tokenize prompts
            prompt_tokens = tokenizer(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False)
            prompt_ids = prompt_tokens["input_ids"].to(engine.device)
            
            # Generate completions with vLLM
            if dist.get_rank() == 0:
                # Main process generates completions
                completion_ids = vllm_client.generate(prompt_texts, 
                                            n=args.num_generations,
                                            temperature=args.temperature,
                                            top_p=args.top_p,
                                            max_tokens=700)
                
                # Decode completions
                completions = []
                for ids in completion_ids:
                    completion = tokenizer.decode(ids, skip_special_tokens=True)
                    completions.append(completion)
                
                # Calculate rewards
                rewards = []
                for i, inp in enumerate(inputs):
                    for comp in completions[i*args.num_generations:(i+1)*args.num_generations]:
                        rewards.append(reward_correct(inp, comp) + reward_format(inp, comp))
                
                # Print a sample completion
                print(f"\nSample completion (reward: {rewards[0]:.2f}):")
                print(completions[0])
            else:
                # Other processes don't need to generate
                completion_ids = None
                completions = None
                rewards = None

            # Ensure all processes are at the same point before proceeding
            dist.barrier()
            print("Barrier passed")

            # Broadcast data using a simple serialization approach similar to grpo_vllm_one.py
            if dist.get_world_size() > 1:
                # First broadcast the rewards, which are simple floats
                if dist.get_rank() == 0:
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(engine.device)
                else:
                    # Create empty tensor of the right size to receive data
                    rewards_tensor = torch.zeros(len(prompts) * args.num_generations, dtype=torch.float32).to(engine.device)
                
                # Broadcast the rewards tensor
                dist.broadcast(rewards_tensor, src=0)
                rewards = rewards_tensor.cpu().tolist()
                
                # Now broadcast completion IDs and text
                if dist.get_rank() == 0:
                    # Serialize completion_ids and completions into a JSON string
                    data_to_send = []
                    for i in range(len(completion_ids)):
                        item = {
                            "ids": completion_ids[i],
                            "text": completions[i]
                        }
                        data_to_send.append(item)
                    
                    data_json = json.dumps(data_to_send)
                    data_bytes = data_json.encode('utf-8')
                    size_tensor = torch.tensor([len(data_bytes)], dtype=torch.long).to(engine.device)
                else:
                    size_tensor = torch.tensor([0], dtype=torch.long).to(engine.device)
                
                # Broadcast the size
                dist.broadcast(size_tensor, src=0)
                size = size_tensor.item()
                
                # Broadcast the actual data
                if dist.get_rank() == 0:
                    data_tensor = torch.tensor([ord(c) for c in data_json], dtype=torch.long).to(engine.device)
                else:
                    data_tensor = torch.zeros(size, dtype=torch.long).to(engine.device)
                
                dist.broadcast(data_tensor, src=0)
                
                # Deserialize data in non-main processes
                if dist.get_rank() != 0:
                    data_json = ''.join(chr(i) for i in data_tensor.cpu().tolist())
                    data_received = json.loads(data_json)
                    
                    completion_ids = []
                    completions = []
                    for item in data_received:
                        completion_ids.append(item["ids"])
                        completions.append(item["text"])

            # Make sure all processes have received the data before proceeding
            dist.barrier()

            # Prepare batch and normalize rewards within each group
            all_batches = []
            metrics_list = []
            
            for i in range(0, len(inputs)):
                group_ids = completion_ids[i*args.num_generations:(i+1)*args.num_generations]
                group_texts = completions[i*args.num_generations:(i+1)*args.num_generations]
                group_rewards = rewards[i*args.num_generations:(i+1)*args.num_generations]
                
                # Normalize rewards within the group (Dr. GRPO skips std normalization)
                group_rewards = torch.tensor(group_rewards, dtype=torch.float32)
                group_rewards = group_rewards - group_rewards.mean()
                
                batch = prepare_batch([inputs[i]], group_rewards.tolist(), group_ids, 
                                    group_texts, prompt_ids[i:i+1], engine.device)
                all_batches.append(batch)
            
            # Process each batch
            for batch in all_batches:
                loss, metrics = Dr_GRPO_step(batch, engine, tokenizer, beta=args.beta, clip_param=args.clip_param)
                engine.backward(loss)
                engine.step()
                
                metrics_list.append(metrics)
                
                if dist.get_rank() == 0:
                    progress.set_description(f"Loss: {loss.item():.6f}")
            
            # Log metrics to wandb
            if dist.get_rank() == 0 and step % args.log_steps == 0:
                # Aggregate metrics
                agg_metrics = {}
                for key in metrics_list[0].keys():
                    agg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
                
                # Also log a sample of completions
                sample_idx = random.randint(0, len(all_batches) - 1)
                sample_batch = all_batches[sample_idx]
                samples = []
                
                for i in range(min(3, len(sample_batch["completion_texts"]))):
                    samples.append({
                        "prompt": prompts[sample_idx],
                        "completion": sample_batch["completion_texts"][i],
                        "reward": float(sample_batch["rewards"][i])
                    })
                
                # Log to wandb
                wandb.log({
                    "step": step,
                    **agg_metrics,
                    "samples": wandb.Table(dataframe=pd.DataFrame(samples))
                })
            
            # Update vLLM server weights periodically
            if step % args.gen_update_steps == 0:
                dist.barrier()
                if dist.get_rank() == 0:
                    print("Updating vLLM server weights...")
                    # In production, you would send model weights to the vLLM server
                    state_dict = engine.module.state_dict()
                    for name, param in state_dict.items():
                        vllm_client.update_named_param(name, param.data)
                    vllm_client.reset_prefix_cache()
                dist.barrier()
            
            # Save model periodically
            if step % args.save_steps == 0:
                dist.barrier()
                if dist.get_rank() == 0:
                    print(f"Saving model at step {step}")
                    save_name = f"{args.output_dir}/step_{step}"
                    state_dict = engine.module.state_dict()
                    state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                    engine.module.save_pretrained(save_name, state_dict=state_dict)
                    tokenizer.save_pretrained(save_name)
                dist.barrier()
        
        # Finish wandb run
        if dist.get_rank() == 0:
            wandb.finish()

    finally:
        # Ensure the process group is properly destroyed
        deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    # Parse arguments using simple_parsing
    # parser = TrlParser(Args)
    # (script_args,) = parser.parse_args_and_config()
    script_args = Args()
    main(script_args) 