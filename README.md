# CS6493_QAsystem
This project is a QA Q&A system, mainly supporting short dialogs, file upload dialogs, and the big prophecy models are chosen to be deployed on ollama with mistral and qwen, so make sure that you have ollama downloaded locally and pull the corresponding models when you use it. The project supports the use of LangChain framework to process the uploaded files and provide them to the model Q&A.

How to use?
1. download Ollama and pull models on Ollama.
   Make sure you have implemented Ollama, run these commands.

   ollama list # You will see the models have been pull.

   ollama pull mistral

   ollama pull qwen2.5:7b

   ollama pull deepseek-r1:7b

   ollama run mistral

   Set environment variable:

   System variable: new->OLLAMA_GPU_LAYER=cuda

2. download the correct software package.

   

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   
   pip install -r requirements.txt
   conda install faiss-gpu==1.7.4
   ```

3. Before run the main function, please run check_env.py and Faiss_test.py. If you get the result like this, that will be fine.

   ```
   [系统信息]
   Python版本: 3.9.21 | packaged by conda-forge | (main, Dec  5 2024, 13:41:22) [MSC v.1929 64 bit (AMD64)]
   numpy版本: 1.26.4
   PyTorch版本: 2.5.1+cu121
   CUDA可用: True
   所用设备：NVIDIA GeForce RTX 4070 Ti SUPER
   
   [关键功能验证]
   D:\conda_envs\QA\lib\site-packages\transformers\utils\generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
     torch.utils._pytree._register_pytree_node(
   ✅ 嵌入模型初始化成功
   向量维度: 384
   ```

   ```
   ✅ FAISS GPU版本工作正常，检测到1块GPU
   ✅ FAISS CPU版本测试通过，搜索结果： [[  0 940 487 737 164]]
   ```

   

4. run main.py