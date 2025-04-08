# Validation scripts
import faiss

def test_faiss():
    # GPU version testing
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(128)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        print(f"✅ FAISS GPU version works fine, {faiss.get_num_gpus()} block GPU detected")

    # CPU version testing
    data = faiss.rand((1000, 128))
    index = faiss.IndexFlatL2(128)
    index.add(data)
    distances, indices = index.search(data[0:1], 5)
    print("✅ FAISS CPU version test passed, search results:", indices)

test_faiss()