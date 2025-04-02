# 验证脚本
import faiss

def test_faiss():
    # GPU版本测试
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(128)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        print(f"✅ FAISS GPU版本工作正常，检测到{faiss.get_num_gpus()}块GPU")

    # CPU版本测试
    data = faiss.rand((1000, 128))
    index = faiss.IndexFlatL2(128)
    index.add(data)
    distances, indices = index.search(data[0:1], 5)
    print("✅ FAISS CPU版本测试通过，搜索结果：", indices)

test_faiss()