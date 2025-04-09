# chunking_analyzer.py
import os
import time
import psutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import subprocess
import gc

class ChunkingAnalyzer:
    """Analyze the impact of different chunking strategies on memory and video memory performance."""
    
    def __init__(self, model_name="mistral", device=None):
        """Initializing the Analyzer

        Args.
            model_name: Ollama model name
            device: device type, automatically detected by default
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        self.output_dir = "./chunking_analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _init_model(self, num_ctx=4096):
        """Initializing the Ollama Model"""
        return Ollama(
            model=self.model_name,
            temperature=0.7,
            num_ctx=num_ctx
        )
    
    def _release_resources(self):
        """Release resources"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        try:
            # Multiple samples for more stable results and peaks
            samples = []
            peak_memory = {}
            utilization_samples = []
            
            for _ in range(5): # Sample 5 times
                # Getting GPU information with the nvidia-smi command
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', 
                     '--format=csv,noheader,nounits'],
                    encoding='utf-8'
                )
                
                # parsing result
                current_sample = {}
                current_utilization = {}
                for i, line in enumerate(result.strip().split('\n')):
                    parts = line.split(', ')
                    name = parts[0]
                    total = float(parts[1].strip())
                    used = float(parts[2].strip())
                    free = float(parts[3].strip())
                    util = float(parts[4].strip()) if len(parts) > 4 else 0
                    
                    # Initialize or update peaks
                    if f"gpu_{i}" not in peak_memory or used > peak_memory[f"gpu_{i}"]:
                        peak_memory[f"gpu_{i}"] = used
                    
                    current_sample[f"gpu_{i}"] = {
                        "device_name": name.strip(),
                        "total_memory_mb": total,
                        "used_memory_mb": used,
                        "free_memory_mb": free,
                    }
                    
                    current_utilization[f"gpu_{i}"] = util
                
                samples.append(current_sample)
                utilization_samples.append(current_utilization)
                time.sleep(0.5)  # Resampling at 0.5 second intervals
            
            # Calculate the average value
            memory_stats = {}
            for device_id in samples[0].keys():
                device_samples = [sample[device_id] for sample in samples]
                util_samples = [sample.get(device_id, 0) for sample in utilization_samples]
                avg_utilization = sum(util_samples) / len(util_samples) if util_samples else 0
                
                memory_stats[device_id] = {
                    "device_name": device_samples[0]["device_name"],
                    "total_memory_mb": device_samples[0]["total_memory_mb"],  # No change in total memory
                    "used_memory_mb": sum(s["used_memory_mb"] for s in device_samples) / len(device_samples),
                    "free_memory_mb": sum(s["free_memory_mb"] for s in device_samples) / len(device_samples),
                    "peak_memory_mb": peak_memory[device_id], 
                    "utilization_percent": round(avg_utilization, 2)  
                }
            
            return {"available": True, "devices": memory_stats}
        except Exception as e:
            # If nvidia-smi is not available, fallback to PyTorch method
            if not torch.cuda.is_available():
                return {"available": False, "message": "CUDA is not available"}
            
            # Get the number of GPUs
            device_count = torch.cuda.device_count()
            memory_stats = {}
            
            for i in range(device_count):
                # Get total and used memory (MB)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                reserved_memory = torch.cuda.memory_reserved(i) / (1024 * 1024)
                allocated_memory = torch.cuda.memory_allocated(i) / (1024 * 1024)
                free_memory = total_memory - reserved_memory
                
                # Get Peak Memory Usage
                try:
                    peak_memory = torch.cuda.max_memory_allocated(i) / (1024 * 1024)
                except:
                    peak_memory = allocated_memory  # If peaks are not available, use the current allocation
                
                memory_stats[f"gpu_{i}"] = {
                    "device_name": torch.cuda.get_device_name(i),
                    "total_memory_mb": round(total_memory, 2),
                    "used_memory_mb": round(allocated_memory, 2),
                    "free_memory_mb": round(free_memory, 2),
                    "peak_memory_mb": round(peak_memory, 2),
                    "utilization_percent": 0  # PyTorch can't get GPU utilization directly
                }
            
            return {"available": True, "devices": memory_stats}
    
    def analyze_chunking_strategy(self, file_path, chunk_sizes=[500, 1000, 1500, 2000], 
                                 chunk_overlaps=[50, 100, 200, 300], query="Please summarize the main points of this document"):
        """Analyzing the performance of different chunking strategies

        Args.
            file_path: document path
            chunk_sizes: list of chunk sizes
            chunk_overlaps: list of chunk overlap sizes
            query: Test query

        Returns: DataFrame: Analysis results
            DataFrame: Analysis results
        """
        results = []
        
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise ValueError(f"File does not exist: {file_path}")
        
        # Get file size (MB)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📄 File Size: {file_size_mb:.2f} MB")
            
        # Selecting a loader based on file type
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(file_path)
        else:
            # Try different coding
            for enc in ['utf-8', 'gbk']:
                try:
                    loader = TextLoader(file_path, encoding=enc)
                    loader.load()  # Test loading
                    break
                except UnicodeDecodeError:
                    continue
        
        # Loading Documents
        load_start = time.time()
        documents = loader.load()
        load_time = time.time() - load_start
        print(f"✅ Successfully loaded documents, total {len(documents)} pages, time {load_time:.2f} seconds")
        
        # Benchmark Memory and Video Memory Usage
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # 转换为MB
        baseline_gpu = self.get_gpu_memory_usage()
        print(f"📊 Baseline Memory Usage. {baseline_memory:.2f} MB")
        
        if baseline_gpu["available"]:
            for device_id, stats in baseline_gpu["devices"].items():
                print(f"📊 GPU {device_id} Benchmark Memory Usage: {stats['used_memory_mb']:.2f} MB / {stats['total_memory_mb']:.2f} MB")
        
        # Testing different chunking strategies
        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                # Skip invalid combinations (overlap greater than block size)
                if chunk_overlap >= chunk_size:
                    continue
                    
                print(f"\n🔍 Test chunking strategy: size={chunk_size}, overlap={chunk_overlap}")
                
                # Release resources
                self._release_resources()
                time.sleep(2)  # Waiting for the system to stabilize
                
                try:
                    # Record memory usage at the start
                    process = psutil.Process(os.getpid())
                    memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
                    gpu_memory_before = self.get_gpu_memory_usage()
                    
                    # Create a Text Splitter
                    start_time = time.time()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", "。", "！", "？", "；", "……", "…", "　"]
                    )
                    
                    # Split Documents
                    docs = text_splitter.split_documents(documents)
                    split_time = time.time() - start_time
                    
                    # Record memory usage after split
                    memory_after_split = process.memory_info().rss / 1024 / 1024
                    memory_split_usage = memory_after_split - memory_before
                    
                    # Create Embeddings
                    embedding_start = time.time()
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-mpnet-base-v2",
                        model_kwargs={"device": self.device},
                        encode_kwargs={"batch_size": 32}
                    )
                    embedding_time = time.time() - embedding_start
                    
                    # Record memory and GPU usage after embedding
                    memory_after_embedding = process.memory_info().rss / 1024 / 1024
                    memory_embedding_usage = memory_after_embedding - memory_after_split
                    gpu_after_embedding = self.get_gpu_memory_usage()
                    
                    # Create a Vector Database
                    vector_db_start = time.time()
                    vector_db = FAISS.from_documents(docs, embeddings)
                    vector_db_time = time.time() - vector_db_start
                    
                    # Record memory usage after vector database creation
                    memory_after_vectordb = process.memory_info().rss / 1024 / 1024
                    memory_vectordb_usage = memory_after_vectordb - memory_after_embedding
                    
                    # Initialize the model
                    model_start = time.time()
                    model = self._init_model()
                    model_init_time = time.time() - model_start
                    
                    # Record GPU usage after model loading
                    gpu_after_model = self.get_gpu_memory_usage()
                    
                    # Create a QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=model,
                        chain_type="stuff",
                        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    
                    # Perform a search
                    query_start = time.time()
                    result = qa_chain({"query": query})
                    query_time = time.time() - query_start
                    
                    # Memory usage at the end of recording
                    memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
                    memory_usage = memory_after - memory_before
                    
                    # Waiting for the system to stabilize
                    time.sleep(2)
                    
                    # Record GPU memory usage
                    gpu_memory_after = self.get_gpu_memory_usage()
                    
                    # Calculate GPU memory changes
                    gpu_memory_diff = {}
                    if gpu_memory_before["available"] and gpu_memory_after["available"]:
                        for device_id, before_stats in gpu_memory_before["devices"].items():
                            after_stats = gpu_memory_after["devices"][device_id]
                            memory_diff = round(after_stats["used_memory_mb"] - before_stats["used_memory_mb"], 2)
                            peak_memory = after_stats.get("peak_memory_mb", after_stats["used_memory_mb"])
                            utilization = after_stats.get("utilization_percent", 0)
                            
                            # Calculate GPU memory increase caused by model loading
                            model_memory_impact = 0
                            if gpu_after_model["available"]:
                                model_stats = gpu_after_model["devices"].get(device_id, {})
                                if model_stats:
                                    model_memory_impact = round(model_stats["used_memory_mb"] - before_stats["used_memory_mb"], 2)
                            
                            # Calculate the increase in GPU memory caused by embedding
                            embedding_memory_impact = 0
                            if gpu_after_embedding["available"]:
                                embedding_stats = gpu_after_embedding["devices"].get(device_id, {})
                                if embedding_stats:
                                    embedding_memory_impact = round(embedding_stats["used_memory_mb"] - before_stats["used_memory_mb"], 2)
                            
                            gpu_memory_diff[device_id] = {
                                "device_name": before_stats["device_name"],
                                "memory_diff_mb": memory_diff,
                                "peak_memory_mb": peak_memory,
                                "utilization_percent": utilization,
                                "model_memory_impact_mb": model_memory_impact,
                                "embedding_memory_impact_mb": embedding_memory_impact
                            }
                    
                    # Calculate average processing time per chunk
                    avg_chunk_process_time = query_time / len(docs) if len(docs) > 0 else 0
                    
                    # Collect results
                    result_entry = {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "num_chunks": len(docs),
                        "avg_chunk_tokens": chunk_size / 4,  # Estimate the number of tokens per chunk
                        "total_tokens": (chunk_size / 4) * len(docs),  # Estimate the total number of tokens
                        "split_time": round(split_time, 2),
                        "embedding_time": round(embedding_time, 2),
                        "vector_db_time": round(vector_db_time, 2),
                        "model_init_time": round(model_init_time, 2),
                        "query_time": round(query_time, 2),
                        "avg_chunk_process_time": round(avg_chunk_process_time, 4),
                        "total_time": round(split_time + embedding_time + vector_db_time + model_init_time + query_time, 2),
                        "ram_usage_mb": round(memory_usage, 2),
                        "ram_split_usage_mb": round(memory_split_usage, 2),
                        "ram_embedding_usage_mb": round(memory_embedding_usage, 2),
                        "ram_vectordb_usage_mb": round(memory_vectordb_usage, 2),
                        "gpu_memory_diff": gpu_memory_diff,
                        "response_length": len(result["result"]),
                        "chunks_per_mb": round(len(docs) / file_size_mb, 2) if file_size_mb > 0 else 0,
                    }
                    
                    if gpu_memory_diff:
                        for device_id, stats in gpu_memory_diff.items():
                            result_entry[f"gpu_{device_id}_diff_mb"] = stats["memory_diff_mb"]
                            result_entry[f"gpu_{device_id}_peak_mb"] = stats["peak_memory_mb"]
                            result_entry[f"gpu_{device_id}_util_percent"] = stats.get("utilization_percent", 0)
                            result_entry[f"gpu_{device_id}_model_impact_mb"] = stats.get("model_memory_impact_mb", 0)
                            result_entry[f"gpu_{device_id}_embedding_impact_mb"] = stats.get("embedding_memory_impact_mb", 0)
                    
                    results.append(result_entry)
                    print(f"✅ Done: chunk_size={chunk_size}, overlap={chunk_overlap}, number of chunks={len(docs)}, total time={result_entry['total_time']} seconds")
                    print(f"   - Memory Usage: {result_entry['ram_usage_mb']:.2f} MB")
                    
                    # 显示GPU内存使用情况
                    if gpu_memory_diff:
                        for device_id, stats in gpu_memory_diff.items():
                            print(f"   - GPU {device_id} GPU memory change: {stats['memory_diff_mb']:.2f} MB, Peak value: {stats['peak_memory_mb']:.2f} MB")
                            print(f"   - GPU {device_id} Utilization: {stats.get('utilization_percent', 0):.2f}%")
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    # Record the error
                    results.append({
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "error": str(e)
                    })
                
                # Release resources
                self._release_resources()
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        return self.results
    
    def visualize_results(self):
        """Visualization of analysis results"""
        if isinstance(self.results, pd.DataFrame):
            if self.results.empty:
                print("No visualization results: DataFrame is empty")
                return None
            
            # Print basic information about the DataFrame to help with diagnostics
            print(f"DataFrame information: {len(self.results)} rows x {len(self.results.columns)} columns")
            print(f"Column name: {list(self.results.columns)}")
            
            # Check if there is an 'error' column, if not then no filtering
            if 'error' in self.results.columns:
                # Filter out results with errors
                valid_results = self.results[~self.results['error'].notna()]
                print(f"Number of valid results after filtering: {len(valid_results)}")
            else:
                # If there is no error column, use all results
                valid_results = self.results
                print("Column 'error' not found, use all results")
            
            if valid_results.empty:
                print("No valid visualization of results after filtering")
                return None
            
            # Create Charts
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            
            # Creating Charts in English
            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.rcParams['axes.unicode_minus'] = True
            
            try:
                # 1. Relationship between chunk size and processing time
                axs[0, 0].set_title('Chunk Size vs Processing Time')
                for overlap in valid_results['chunk_overlap'].unique():
                    subset = valid_results[valid_results['chunk_overlap'] == overlap]
                    axs[0, 0].plot(subset['chunk_size'], subset['total_time'], 'o-', label=f'overlap={overlap}')
                axs[0, 0].set_xlabel('Chunk size')
                axs[0, 0].set_ylabel('Total Time (s)')
                axs[0, 0].legend()
                axs[0, 0].grid(True)
                
                # 2. Relationship between chunk size and memory usage
                axs[0, 1].set_title('Chunk Size vs Memory Usage')
                for overlap in valid_results['chunk_overlap'].unique():
                    subset = valid_results[valid_results['chunk_overlap'] == overlap]
                    axs[0, 1].plot(subset['chunk_size'], subset['ram_usage_mb'], 'o-', label=f'overlap={overlap}')
                axs[0, 1].set_xlabel('Chunk Size')
                axs[0, 1].set_ylabel('RAM Usage (MB)')
                axs[0, 1].legend()
                axs[0, 1].grid(True)
                
                # 3. Relationship between number of chunks and processing time
                axs[1, 0].set_title('Number of Chunks vs Processing Time')
                axs[1, 0].scatter(valid_results['num_chunks'], valid_results['total_time'], 
                                 c=valid_results['chunk_size'], cmap='viridis', alpha=0.7)
                axs[1, 0].set_xlabel('Number of Chunks')
                axs[1, 0].set_ylabel('Total Time (s)')
                axs[1, 0].grid(True)
                
                # 4. GPU memory usage (if any)
                axs[1, 1].set_title('Chunk Size vs GPU Memory Usage')
                gpu_cols = [col for col in valid_results.columns if 'gpu_' in col and 'diff_mb' in col]
                
                if gpu_cols:
                    for gpu_col in gpu_cols:
                        device_id = gpu_col.split('_')[1]
                        for overlap in valid_results['chunk_overlap'].unique():
                            subset = valid_results[valid_results['chunk_overlap'] == overlap]
                            if gpu_col in subset.columns:
                                axs[1, 1].plot(subset['chunk_size'], subset[gpu_col], 'o-', 
                                              label=f'GPU {device_id}, overlap={overlap}')
                    axs[1, 1].set_xlabel('Chunk Size')
                    axs[1, 1].set_ylabel('GPU Memory Change (MB)')
                    axs[1, 1].legend()
                    axs[1, 1].grid(True)
                else:
                    axs[1, 1].text(0.5, 0.5, 'GPU Data Not Available', ha='center', va='center', fontsize=14)
                    axs[1, 1].axis('off')
                
                plt.tight_layout()
                
                # Save chart
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_dir, f"chunking_analysis_{timestamp}.png")
                plt.savefig(output_path)
                print(f"✅ Charts have been saved to: {output_path}")
                
                # Show chart
                plt.show()
                
                return output_path
                
            except Exception as e:
                print(f"❌ Errors during visualization: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print(f"❌ The result is not in DataFrame format: {type(self.results)}")
            return None
    
    def export_results(self):
        """Export analysis results to Excel"""
        if self.results.empty:
            print("No results to export")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"chunking_analysis_{timestamp}.xlsx")
        
        # Export to Excel
        self.results.to_excel(output_path, index=False)
        print(f"✅ Results have been exported to: {output_path}")
        
        return output_path


# usage example
def run_analysis(file_path, model_name="mistral", 
                chunk_sizes=[500, 1000, 1500, 2000], 
                chunk_overlaps=[50, 100, 200]):
    """Running a chunking strategy analysis

    Args.
        file_path: file path
        model_name: Ollama model name
        chunk_sizes: list of chunk sizes to be tested
        chunk_overlaps: list of chunk overlaps to test
    """
    analyzer = ChunkingAnalyzer(model_name=model_name)
    
    print(f"🔍 Start analyzing the file: {file_path}")
    print(f"📊 Test chunk size: {chunk_sizes}")
    print(f"📊 Test overlap size: {chunk_overlaps}")
    
    # Run analysis
    results = analyzer.analyze_chunking_strategy(
        file_path=file_path,
        chunk_sizes=chunk_sizes,
        chunk_overlaps=chunk_overlaps
    )
    
    # Export results
    analyzer.export_results()
    
    # Visualization
    analyzer.visualize_results()
    
    return results


if __name__ == "__main__":
    # It won't be implemented here because it's a standalone analysis tool
    pass