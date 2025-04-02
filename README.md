# CS6493_QAsystem
This project is a QA Q&A system, mainly supporting short dialogs, file upload dialogs, and the big prophecy models are chosen to be deployed on ollama with mistral and qwen, so make sure that you have ollama downloaded locally and pull the corresponding models when you use it. The project supports the use of LangChain framework to process the uploaded files and provide them to the model Q&A.

How to use?
1. download Ollama and pull models on Ollama.
   Make sure you have implemented Ollama, run these commands.
   
   ollama list # You will see the models have been pull.
   
   ollama pull mistral
   
   ollama pull qwen:1.8b
   
   ollama run mistral
   
3. download the correct software package.
   
   ```bash
   pip install -r requirements.txt
   ```
   
4. Before run the main function, please run check_env.py. If you get the result like this, that will be fine.
   
   ```
   [系统信息]
   Python版本: 3.9.21 | packaged by conda-forge | (main, Dec  5 2024, 13:41:22) [MSC v.1929 64 bit (AMD64)]
   numpy版本: 1.23.5
   PyTorch版本: 2.0.1+cpu
   CUDA可用: False
   
   [关键功能验证]
   ✅ 嵌入模型初始化成功
   向量维度: 384
   ```
   
   
   
5. run main.py