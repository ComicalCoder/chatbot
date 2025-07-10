import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_mistral():
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ.get("HF_HOME"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,
        device_map="auto",
        cache_dir=os.environ.get("HF_HOME"),
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    # Debug: Print parameter devices and dtypes
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on {param.device} with dtype {param.dtype}")
    return tokenizer, model

def generate_response(tokenizer, model, prompt):
    # Ensure inputs are on the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("Loading Mixtral-8x7B-Instruct-v0.1 model... (This may take a moment)")
    tokenizer, model = load_mistral()
    print("Model loaded! Type your question or 'exit' to quit.")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        response = generate_response(tokenizer, model, prompt)
        print(f"Mixtral: {response}")

if __name__ == "__main__":
    main()