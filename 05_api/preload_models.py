import os
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

def preload():
    print("Pre-downloading Qwen2.5-7B-Instruct-GPTQ-Int4...")
    snapshot_download("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf"])
    
    print("Pre-downloading BAAI/bge-large-en-v1.5...")
    snapshot_download("BAAI/bge-large-en-v1.5")
    
    print("Pre-downloading cross-encoder/ms-marco-MiniLM-L-6-v2...")
    snapshot_download("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    print("All models successfully cached!")

if __name__ == "__main__":
    preload()
