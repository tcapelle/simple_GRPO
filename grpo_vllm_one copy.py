from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import wandb  # Import wandb
import traceback
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "./data/Qwen2.5-3B"
gen_device = 1    # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
beta = 0.04
all_steps = 1000
Q_batch_size = 5
num_pre_Q = 8
train_batch_size = 4
gen_update_steps = 8
save_steps = 200
compute_gen_logps = True
clip_param = 0.2
ref_server = "http://localhost:59875"
wandb_project = "grpo_simple"  # Define wandb project name
wandb_run_name = None
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
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

def get_batch():
    try:
        print("DEBUG: Requesting batch from server...")
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': 
            print("DEBUG: Server returned empty response")
            return None
        print(f"DEBUG: Server returned data, size: {len(r)} bytes")
    except Exception as e:
        print(f"ERROR in get_batch: {e}")
        return None
    
    try:
        dd = bytes_list_to_list(r)
        data = json.loads(dd[0]) 
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        data['refs'] = bytes_to_tensor(dd[3])
        if len(dd) == 5: data['gen_logps'] = bytes_to_tensor(dd[4])
        print(f"DEBUG: Successfully parsed batch, shape: {data['inputs'].shape}")
        return data
    except Exception as e:
        print(f"ERROR parsing batch data: {e}")
        return None

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather

def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # Calculate metrics for wandb
    metrics = {
        "loss": loss.item(),
        "kl_penalty": (beta * per_token_kl * completion_mask).sum().item() / completion_mask.sum().item(),
    }
    if 'gen_logps' in batch:
        metrics["policy_ratio_mean"] = ratio.mean().item()
        metrics["policy_ratio_max"] = ratio.max().item()
        metrics["policy_ratio_min"] = ratio.min().item()
    
    return loss, metrics


def gen_worker(Q, physics_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print("\n" + "="*50)
    print("vLLM WORKER STARTING")
    print("GPU Device:", physics_device)
    print("="*50 + "\n")
    print(f"Generation worker process uses GPU {physics_device}")
    
    try:
        from vllm import LLM, SamplingParams
        print("Successfully imported vLLM")
        
        vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.8)
        print("Successfully loaded vLLM model")
        
        ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

        sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=700)
        gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

        try:
            print("Attempting to load GSM8K dataset...")
            dataset = load_dataset("openai/gsm8k", "main", split="train")
            print(f"Successfully loaded dataset with {len(dataset)} examples")
            QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
            print(f"Processed {len(QAs)} QA pairs")
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            return
            
        try:
            print("Testing server connection before starting...")
            test_response = requests.get(f"{ref_server}/get")
            print(f"Server connection test: {test_response.status_code}, content: {test_response.content}")
        except Exception as e:
            print(f"ERROR connecting to server: {e}")
            
        system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
        The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
        
        def gen_answers(prompts):
            print(f"Generating answers for {len(prompts)} prompts")
            tip_text = []
            for x in prompts:
                tip_text.append(tokenizer.apply_chat_template([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
            voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
            answers = [];  ans_token_ids = []
            for v in voutputs:
                for z in v.outputs: 
                    answers.append(z.text)
                    ans_token_ids.append(z.token_ids)
            print(f"Generated {len(answers)} answers")
            return answers, ans_token_ids

        try:
            print("Attempting to import math_verify...")
            from math_verify import parse, verify, ExprExtractionConfig
            print("Successfully imported math_verify")
        except Exception as e:
            print(f"ERROR importing math_verify: {e}")
            
            # Provide fallback implementations if math_verify is missing
            print("Using fallback implementations for parse and verify")
            def parse(text, extraction_config=None):
                try:
                    return float(text.strip())
                except:
                    return 0.0
                
            def verify(ans, ground_truth):
                return abs(ans - ground_truth) < 1e-6
                
            class ExprExtractionConfig:
                pass

        def reward_correct(item, answer):
            pattern = r'\d+\.\d+|\d+/\d+|\d+'
            nums = re.findall(pattern, answer) 
            if len(nums) == 0: return -1.0
            lastnum = nums[-1]
            ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
            ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
            return 1 if verify(ans, ground_truth) else -1
        def reward_format(item, answer):
            pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
            think_count = answer.count("<think>") + answer.count("</think>")
            answer_count = answer.count("<answer>") + answer.count("</answer>")
            return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2 else -1


        def gen_samples(inputs):
            prompts = [x["Q"] for x in inputs]
            answers, ans_token_ids = gen_answers(prompts)
            rewards = []
            for i, inp in enumerate(inputs):
                for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                    rewards.append(reward_correct(inp, a) + reward_format(inp, a))
            prompts_text = [tokenizer.apply_chat_template([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]         
            return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids

        def try_update_model():
            try:
                new_state_dict = Q.get_nowait()
                print('[VLLM PROC] recving new model ...')
                llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(new_state_dict.items())
                print('[VLLM PROC] model updated')
                del new_state_dict
            except:
                #print('[VLLM PROC] no new model')
                return
        
        from torch.nn.utils.rnn import pad_sequence
        for it in range(999999999):
            try:
                print(f"\n[ITER {it}] Starting generation iteration")
                if it % 3 == 0: 
                    print("[ITER %d] Checking for model updates" % it)
                    try_update_model()
                    
                print(f"[ITER {it}] Sampling {Q_batch_size} questions from {len(QAs)} QAs")
                inputs = random.sample(QAs, Q_batch_size)
                print(f"[ITER {it}] Starting gen_samples on {len(inputs)} inputs")
                tic = time.time()
                prompt_inputs, rewards, answers, ans_token_ids = gen_samples(inputs)
                print(f'[ITER {it}] time: {time.time()-tic:.2f}s    ', 'rewards:', rewards, )
                if it % 5 == 0: print('[ITER %d] answers:' % it, answers[0])

                print(f"[ITER {it}] Processing {len(prompt_inputs)} prompts")
                for i, pp in enumerate(prompt_inputs):
                    print(f"[ITER {it}] Processing prompt {i+1}/{len(prompt_inputs)}")
                    prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
                    plen = prompt_ids.shape[1]
                    curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
                    curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
                    curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
                    
                    if curr_rewards.max() - curr_rewards.min() < 1e-4: 
                        print(f"[ITER {it}] Skipping prompt {i+1} due to uniform rewards")
                        continue

                    print(f"[ITER {it}] Using {ref_server_ver} mode for upload")
                    if ref_server_ver == 'tensor':
                        curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                        for ii in range(0, num_pre_Q, train_batch_size):
                            print(f"[ITER {it}] Processing batch {ii//train_batch_size + 1}/{(num_pre_Q+train_batch_size-1)//train_batch_size}")
                            sub_rewards = curr_rewards[ii:ii+train_batch_size]
                            sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                            print(f"[ITER {it}] Creating tensor list from {len(sub_ans_ids)} answer IDs")
                            tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                            output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                            Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                            merged_ids = torch.cat([Qrep, output_ids], dim=1)
                            data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]       

                            if compute_gen_logps:
                                print(f"[ITER {it}] Computing generation logps")
                                zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                                zz = [xx.prompt_logprobs[plen:] for xx in zz]
                                gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                                data.append(tensor_to_bytes(gen_logps))

                            xdata = make_bytes_list(data)
                            try:
                                print(f"[ITER {it}] Attempting to upload data to {ref_server}/upload, data size: {len(xdata)} bytes")
                                r = requests.post(f"{ref_server}/upload", data=xdata)
                                print(f"[ITER {it}] Upload response: {r.content}, status: {r.status_code}")
                                if r.content == b'string': ref_server_ver = 'string'
                            except Exception as e:
                                print(f"[ITER {it}] ERROR uploading to server: {e}")
                    elif ref_server_ver == 'string':
                        xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                                tensor_to_bytes(curr_rewards)])
                        try:
                            print(f"[ITER {it}] Attempting string upload to {ref_server}/upload")
                            r = requests.post(f"{ref_server}/upload", data=xdata)
                            print(f"[ITER {it}] String upload response: {r.content}")
                            if r.content == b'tensor': ref_server_ver = 'tensor'
                        except Exception as e:
                            print(f"[ITER {it}] ERROR with string upload: {e}")
            except Exception as e:
                print(f"ERROR in iteration {it}: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"CRITICAL ERROR in gen_worker: {e}")
        traceback.print_exc()
        return

def check_server_connection():
    try:
        r = requests.get(f"{ref_server}/get", timeout=5)
        print(f"Server connection check: status code {r.status_code}")
        return True
    except Exception as e:
        print(f"Server connection failed: {str(e)}")
        return False

tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()

    # Initialize wandb only on rank 0
    if dist.get_rank() == 0:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model": model_path,
                "beta": beta,
                "steps": all_steps,
                "Q_batch_size": Q_batch_size,
                "num_pre_Q": num_pre_Q,
                "train_batch_size": train_batch_size,
                "gen_update_steps": gen_update_steps,
                "clip_param": clip_param,
                "compute_gen_logps": compute_gen_logps,
            }
        )
        
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, gen_device))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                model_parameters=model.parameters())
    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: progress = tqdm(progress)
    for step in progress:
        batch = get_batch()
        while batch is None:
            print('waiting for batch...'); time.sleep(1)
            batch = get_batch()

        loss, metrics = GRPO_step(batch)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")
            # Log metrics to wandb
            wandb_metrics = {
                "train/loss": metrics["loss"],
                "train/kl_penalty": metrics["kl_penalty"],
                "step": step,
            }
            
            if "policy_ratio_mean" in metrics:
                wandb_metrics["train/policy_ratio_mean"] = metrics["policy_ratio_mean"]
                wandb_metrics["train/policy_ratio_max"] = metrics["policy_ratio_max"]
                wandb_metrics["train/policy_ratio_min"] = metrics["policy_ratio_min"]
                

            wandb.log(wandb_metrics)

        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
            dist.barrier()

        if step % save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"./step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
                # Save model artifact to wandb
                artifact = wandb.Artifact(f'model-step-{step}', type='model')
                artifact.add_dir(save_name)
                wandb.log_artifact(artifact)
            dist.barrier()

        if dist.get_rank() == 0:
            server_reachable = check_server_connection()
            if not server_reachable:
                print("WARNING: Cannot connect to reference server. Check your network settings.")
                # You might want to exit here if the server is essential
    
    # Finish wandb run
    if dist.get_rank() == 0:
        wandb.finish()
