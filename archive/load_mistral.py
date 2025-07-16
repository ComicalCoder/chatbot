import os
from transformers import pipeline, BitsAndBytesConfig
import torch

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model using pipeline with quantization
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
chatbot = pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"quantization_config": quant_config, "device_map": "cuda", "torch_dtype": torch.float16},
    tokenizer=model_name
)
chatbot.tokenizer.pad_token = chatbot.tokenizer.eos_token  # Set pad token to EOS token

# Manually configurable parameters
temperature = 0.7  # Controls randomness (0.1-1.5)
top_p = 0.95  # Nucleus sampling (0.1-1.0)
top_k = 50  # Top-k sampling (10-100)
max_new_tokens = 256  # Response length (50-500)
system_message = "You are a professional office worker providing clear and concise responses in a formal tone."

# Function to generate response
def generate_response(messages):
    outputs = chatbot(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=chatbot.tokenizer.eos_token_id,
        do_sample=True  # Enable sampling for varied responses
    )
    response = outputs[0]["generated_text"][-1]["content"]  # Extract the last message's content
    return response

# Main interactive loop
def main():
    print("Model loaded! Type your question or 'exit' to quit.")
    messages = [{"role": "system", "content": system_message}]

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        else:
            messages.append({"role": "user", "content": prompt})
            response = generate_response(messages)
            messages.append({"role": "assistant", "content": response})
            print(f"Mixtral: {response}")

if __name__ == "__main__":
    main()