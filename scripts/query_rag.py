import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_mistral():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ.get("HF_HOME"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        cache_dir=os.environ.get("HF_HOME"),
        low_cpu_mem_usage=True
    )
    return tokenizer, model

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index"), embeddings, allow_dangerous_deserialization=True)

def query_rag(query, tokenizer, model, vector_store, top_k=3):
    docs = vector_store.similarity_search(query, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""Jawab soalan berikut berdasarkan konteks yang diberikan. Jawab dalam bahasa yang sama dengan soalan (Bahasa Inggeris, Bahasa Melayu, atau campuran). Jika konteks tidak memberikan maklumat yang cukup, nyatakan sedemikian.

Konteks:
{context}

Soalan:
{query}

Jawapan:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r"Jawapan:\s*(.*?)(?:\n|$)", response, re.DOTALL)
    answer = match.group(1).strip() if match else response.strip()
    return answer, [doc.metadata["source"] for doc in docs]

def main():
    tokenizer, model = load_mistral()
    vector_store = load_vector_store()
    queries = [
        "What services does JPKN Sabah provide?",
        "Apa perkhidmatan yang ditawarkan oleh JPKN?",
        "JPKN Sabah services apa?"
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        response, sources = query_rag(query, tokenizer, model, vector_store)
        print(f"Response: {response}")
        print(f"Sources: {sources}")

if __name__ == "__main__":
    main()