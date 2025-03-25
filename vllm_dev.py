from vllm import LLM, SamplingParams

llm = LLM(
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    revision="main",
)

sampling_params = SamplingParams(
    max_tokens=10,
)

out = llm.generate(
    "Hello, how are you?",
    sampling_params=sampling_params,
)

print(out)