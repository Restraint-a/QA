# models.py

import torch
import subprocess
import time
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

class DocumentQASystem:
    def __init__(self):
        self.llm_registry = self._init_models()
        self.current_model = "mistral"
        self.memory = ConversationBufferMemory()
        self.vector_db = None
        self.qa_chain = None
        self.conversation_chain = None

    def _init_models(self):
        """初始化模型实例并缓存"""
        return {
            "mistral": Ollama(
                model="mistral",
                temperature=0.7,
                num_ctx=4096,
            ),
            "qwen": Ollama(
                model="qwen2.5:7b",
                temperature=0.7,
                num_ctx=4096,
            ),
            "deepseek": Ollama(
                model="deepseek-r1:7b",
                temperature=0.7,
                num_ctx=4096,
            ),
            "gemma3": Ollama(
                model="gemma3:4b",
                temperature=0.7,
                num_ctx=4096,
            ),
        }

    def _release_model_resources(self):
        """彻底释放模型资源"""
        for model in self.llm_registry.values():
            if hasattr(model, 'client'):
                del model.client
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_gpu_memory_usage(self):
        """获取当前GPU显存使用情况"""
        try:
            
            # 多次采样以获取更稳定的结果和峰值
            samples = []
            peak_memory = {}
            
            for _ in range(3):  # 采样3次
                # 使用nvidia-smi命令获取GPU信息 - 移除utilization.gpu参数
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', 
                     '--format=csv,noheader,nounits'],
                    encoding='utf-8'
                )
                
                # 解析结果
                current_sample = {}
                for i, line in enumerate(result.strip().split('\n')):
                    parts = line.split(', ')
                    name = parts[0]
                    total = float(parts[1].strip())
                    used = float(parts[2].strip())
                    free = float(parts[3].strip())
                    
                    # 初始化或更新峰值
                    if f"gpu_{i}" not in peak_memory or used > peak_memory[f"gpu_{i}"]:
                        peak_memory[f"gpu_{i}"] = used
                    
                    current_sample[f"gpu_{i}"] = {
                        "device_name": name.strip(),
                        "total_memory_mb": total,
                        "used_memory_mb": used,
                        "free_memory_mb": free,
                    }
                samples.append(current_sample)
                time.sleep(0.5)  # 间隔0.5秒再次采样
            
            # 计算平均值
            memory_stats = {}
            for device_id in samples[0].keys():
                device_samples = [sample[device_id] for sample in samples]
                memory_stats[device_id] = {
                    "device_name": device_samples[0]["device_name"],
                    "total_memory_mb": device_samples[0]["total_memory_mb"],  # 总内存不变
                    "used_memory_mb": sum(s["used_memory_mb"] for s in device_samples) / len(device_samples),
                    "free_memory_mb": sum(s["free_memory_mb"] for s in device_samples) / len(device_samples),
                    "peak_memory_mb": peak_memory[device_id],  # 添加峰值
                }
            
            return {"available": True, "devices": memory_stats}
        except Exception as e:
            # 如果nvidia-smi不可用，回退到PyTorch方法
            if not torch.cuda.is_available():
                return {"available": False, "message": "CUDA不可用"}
            
            # 获取GPU数量
            device_count = torch.cuda.device_count()
            memory_stats = {}
            
            for i in range(device_count):
                # 获取总显存和已用显存(MB)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                reserved_memory = torch.cuda.memory_reserved(i) / (1024 * 1024)
                allocated_memory = torch.cuda.memory_allocated(i) / (1024 * 1024)
                free_memory = total_memory - reserved_memory
                
                memory_stats[f"gpu_{i}"] = {
                    "device_name": torch.cuda.get_device_name(i),
                    "total_memory_mb": round(total_memory, 2),
                    "used_memory_mb": round(allocated_memory, 2),
                    "free_memory_mb": round(free_memory, 2),
                    "peak_memory_mb": round(allocated_memory, 2),  # PyTorch方式下使用当前分配作为峰值
                }
            
            return {"available": True, "devices": memory_stats}