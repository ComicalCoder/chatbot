import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_mistral():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ.get("HF_HOME"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=os.environ.get("HF_HOME"),
        low_cpu_mem_usage=True

    )
    return tokenizer, model

def test_mistral():
    tokenizer, model = load_mistral()
    prompt = "Hello, how can I assist you today?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test response: {response}")

if __name__ == "__main__":
    test_mistral()