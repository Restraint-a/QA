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
        """Initialize model instances and cache"""
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
            "gemma3": Ollama(
                model="gemma3:4b",
                temperature=0.7,
                num_ctx=4096,
            ),
        }

    def _release_model_resources(self):
        """Complete release of model resources"""
        for model in self.llm_registry.values():
            if hasattr(model, 'client'):
                del model.client
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        try:
            
            # Multiple samples for more stable results and peaks
            samples = []
            peak_memory = {}
            
            for _ in range(3):  # Sampling 3 times
                # Getting GPU information with the nvidia-smi command
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', 
                     '--format=csv,noheader,nounits'],
                    encoding='utf-8'
                )
                
                # parse result
                current_sample = {}
                for i, line in enumerate(result.strip().split('\n')):
                    parts = line.split(', ')
                    name = parts[0]
                    total = float(parts[1].strip())
                    used = float(parts[2].strip())
                    free = float(parts[3].strip())
                    
                    # Initialize or update peak values
                    if f"gpu_{i}" not in peak_memory or used > peak_memory[f"gpu_{i}"]:
                        peak_memory[f"gpu_{i}"] = used
                    
                    current_sample[f"gpu_{i}"] = {
                        "device_name": name.strip(),
                        "total_memory_mb": total,
                        "used_memory_mb": used,
                        "free_memory_mb": free,
                    }
                samples.append(current_sample)
                time.sleep(0.5)  # Re-sampling after 0.5 second intervals
            
            # Calculation of average values
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
            # If nvidia-smi is not available, fallback to PyTorch method
            if not torch.cuda.is_available():
                return {"available": False, "message": "CUDA不可用"}
            
            # Get the number of GPUs
            device_count = torch.cuda.device_count()
            memory_stats = {}
            
            for i in range(device_count):
                # Get total memory and used memory (MB)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                reserved_memory = torch.cuda.memory_reserved(i) / (1024 * 1024)
                allocated_memory = torch.cuda.memory_allocated(i) / (1024 * 1024)
                free_memory = total_memory - reserved_memory
                
                memory_stats[f"gpu_{i}"] = {
                    "device_name": torch.cuda.get_device_name(i),
                    "total_memory_mb": round(total_memory, 2),
                    "used_memory_mb": round(allocated_memory, 2),
                    "free_memory_mb": round(free_memory, 2),
                    "peak_memory_mb": round(allocated_memory, 2),  # The PyTorch approach uses the current allocation as a peak
                }
            
            return {"available": True, "devices": memory_stats}