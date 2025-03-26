from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import re
import wandb
from math_verify import parse, verify, ExprExtractionConfig
from datasets import load_dataset, Dataset


def reward_correct(completions, answer, **kwargs):
    """Verify if the completions is mathematically correct"""
    responses = [completion[0]['content'] for completion in completions]
    def _reward_correct(one_response, one_answer):
        pattern = r"\d+\.\d+|\d+/\d+|\d+"
        nums = re.findall(pattern, one_response)
        if len(nums) == 0:
            return -1.0
        lastnum = nums[-1]
        try:
            ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
            ground_truth = parse(one_answer, extraction_config=[ExprExtractionConfig()])
            return 1.0 if verify(ans, ground_truth) else -1.0
        except:
            return -1.0
    return [_reward_correct(response, answer) for response, answer in zip(responses, answer)]

def reward_format(completions, **kwargs):
    """Verify if the completions follow the required format"""
    responses = [completion[0]['content'] for completion in completions]
    def _reward_format(one_response):
        pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
        think_count = one_response.count("<think>") + one_response.count("</think>")
        answer_count = one_response.count("<answer>") + one_response.count("</answer>")
        return (
            1.25
            if re.match(pattern, one_response, re.DOTALL | re.VERBOSE)
            and think_count == 2
            and answer_count == 2
            else -1.0
        )
    return [_reward_format(response) for response in responses]


model = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model)


# Load and prep dataset
SYSTEM_PROMPT = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


XML_COT_FORMAT = """\
<think>
{reasoning}
</think>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

training_args = GRPOConfig(
    use_vllm = False, # use vLLM for fast inference!
    model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "cuda:0",
    },
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    beta=0.0,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    log_completions = True,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "grpo_trl_output",
)

wandb.init(project="grpo-trl", config=training_args)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [reward_correct, reward_format],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()