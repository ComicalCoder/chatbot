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

    echo "Attempt $attempt of $max_attempts: Installing model (4-bit quantized with CPU offload, nf4)..."
    python_output=$(python -c "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig; import torch; config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True, bnb_4bit_quant_type='nf4'); model = AutoModelForCausalLM.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1', quantization_config=config, device_map='auto', resume_download=True); tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1'); print('Model loaded successfully')" 2>&1)
    echo "$python_output"
    
    if echo "$python_output" | grep -q "Model loaded successfully"; then
        echo "Model installed successfully on attempt $attempt!"
        exit 0
    else
        echo "Attempt $attempt failed. Retrying..."
        ((attempt++))
        sleep 15
    fi
done

echo "Failed to install model after $max_attempts attempts. Consider checking hardware or trying 8-bit with CPU offload and nf4."
exit 1