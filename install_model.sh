#!/bin/bash

max_attempts=200
attempt=1
min_space=500000000000  # 500GB in bytes (500 * 1024^3)

while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt of $max_attempts: Checking disk space..."
    available_space=$(df -B1 --output=avail / | tail -n 1)

    if [ "$available_space" -lt "$min_space" ]; then
        echo "Disk space below 500GB ($available_space bytes available). Stopping to prevent memory overload."
        exit 1
    fi

    echo "Attempt $attempt of $max_attempts: Installing Mistral-7B v0.3 (4-bit quantized)..." # <--- Updated message
    # Enable hf_transfer for robust downloads
    export HF_HUB_ENABLE_HF_TRANSFER=1

    # --- IMPORTANT CHANGES IN THE PYTHON -c COMMAND BELOW ---
    # Changed model name to v0.3
    python_output=$(python -c "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig; import torch; config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16); model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3', quantization_config=config, device_map='auto', resume_download=True, torch_dtype=torch.float16); tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3'); print('Model loaded successfully')" 2>&1)
    # --- END OF IMPORTANT CHANGES ---

    echo "$python_output"

    if echo "$python_output" | grep -q "Model loaded successfully"; then
        echo "Model installed successfully on attempt $attempt!"
        exit 0
    else
        echo "Attempt $attempt failed. Retrying..."
        ((attempt++))
        sleep 30
    fi
done

echo "Failed to install model after $max_attempts attempts. Consider checking hardware."
exit 1