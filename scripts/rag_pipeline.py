import os
import re
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress transformers and other noisy logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_mistral():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offload
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ.get("HF_HOME"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",  # Auto-distribute across GPU and CPU
        cache_dir=os.environ.get("HF_HOME"),
        low_cpu_mem_usage=True
    )
    return tokenizer, model

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.load_local(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index"), embeddings, allow_dangerous_deserialization=True)
    # Debug: List all sources in vectorstore
    sources = set(doc.metadata["source"] for doc in vector_store.docstore._dict.values())
    logger.info(f"Loaded vectorstore with sources: {sorted(sources)}")
    return vector_store

def query_rag(query, tokenizer, model, vector_store, top_k=7):
    docs = vector_store.similarity_search(query, k=top_k)
    # Prioritize sources
    prioritized_docs = []
    for doc in docs:
        if "director" in query.lower() or "pengarah" in query.lower():
            if "direktori" in doc.metadata["source"]:  # Match direktori_.json or https://jpkn.sabah.gov.my/direktori/
                prioritized_docs.insert(0, doc)
            elif "168-2" in doc.metadata["source"]:
                prioritized_docs.insert(len(prioritized_docs)//2, doc)
            else:
                prioritized_docs.append(doc)
        elif "JPAN_700-28(44)" in query:
            if "JPAN_700-28(44)" in doc.metadata["source"]:
                prioritized_docs.insert(0, doc)
            else:
                prioritized_docs.append(doc)
        elif "working hours" in query.lower() or "masa bekerja" in query.lower():
            if "187-2" in doc.metadata["source"]:
                prioritized_docs.insert(0, doc)
            else:
                prioritized_docs.append(doc)
        else:
            prioritized_docs.append(doc)
    docs = prioritized_docs[:top_k]
    context = "\n".join([doc.page_content for doc in docs])
    logger.info(f"Query: {query}, Retrieved sources: {[doc.metadata['source'] for doc in docs]}")
    # Determine response language
    is_malay = any(word in query.lower() for word in ["siapakah", "apa", "masa bekerja"])
    prompt = f"""Answer the following question based on the provided context. 
    Respond in {'Malay' if is_malay else 'English'}, matching the language of the question. 
    Provide a concise, natural response without mentioning sources or technical details like JSON files or URLs. 
    Ensure consistency across languages.
    Expect sources to be in mostly in Malay. 
    Do translation from English to Malay to the prompts to easier find data.
    

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r"Answer:\s*(.*?)(?:\n|$)", response, re.DOTALL)
    answer = match.group(1).strip() if match else response.strip()
    return answer

def main():
    try:
        tokenizer, model = load_mistral()
        vector_store = load_vector_store()
        queries = [
            "Who's the director of JPKN ?",
            "Siapakah pengarah JPKN ?",
            "Apa masa bekerja JPKN ?",
            "What are JPKN's working hours ?",
            "What is in JPAN_700-28(44).pdf?"
        ]
        for query in queries:
            print(f"\nQuery: {query}")
            response = query_rag(query, tokenizer, model, vector_store)
            print(f"Response: {response}")
    except Exception as e:
        print(f"[ERROR] Failed to run query: {e}")

if __name__ == "__main__":
    main()