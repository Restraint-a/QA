# check_env.py
import sys, torch, numpy
from langchain_community.embeddings import HuggingFaceEmbeddings

print("[System Info]")
print(f"Python version: {sys.version}")
print(f"numpy version: {numpy.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device used:{torch.cuda.get_device_name(0)}")

print("\n[Critical Function Verification]")

try:
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("✅ Embedded model initialization successful")
    test_text = "Test Chinese vectorization"
    embedding = emb.embed_query(test_text)
    print(f"Vector Dimension: {len(embedding)}")
except Exception as e:
    print(f"❌ Initialization fail: {str(e)}")