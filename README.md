# CS6493_QAsystem
This project is a QA Q&A system, mainly supporting short dialogs, file upload dialogs, and the big prophecy models are chosen to be deployed on ollama with mistral and qwen, so make sure that you have ollama downloaded locally and pull the corresponding models when you use it. The project supports the use of LangChain framework to process the uploaded files and provide them to the model Q&A.

How to use?
1. download Ollama and pull models on Ollama.
   Make sure you have implemented Ollama, run these commands.

   ollama list # You will see the models have been pull.

   ATTENTION: Choose the models according to your gpu memory.

   ollama pull mistral

   ollama pull qwen2.5:7b

   ollama pull deepseek-r1:7b

   ollama pull gemma3:4b

   ollama run mistral

   Set environment variable:

   System variable: new->OLLAMA_GPU_LAYER=cuda

3. download the correct software package.

   Please create the env on conda since it seems that faiss-gpu packet only can be install by 'conda install' command.

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   
   pip install -r requirements.txt
   conda install faiss-gpu==1.7.4
   ```

4. Before run the main function, please run check_env.py and Faiss_test.py. If you get the result like this, that will be fine.

   ```
   [System Info]
   Python version: 3.9.21 | packaged by conda-forge | (main, Dec  5 2024, 13:41:22) [MSC v.1929 64 bit (AMD64)]
   numpy version: 1.23.5
   PyTorch version: 2.5.1+cu121
   CUDA available: True
   Device used:NVIDIA GeForce RTX 4070 Ti SUPER
   
   [Critical Function Verification]
   ✅ Embedded model initialization successful
   Vector Dimension: 384
   ```
   
   ```
✅ FAISS GPU version works fine, 1 block GPU detected
   ✅ FAISS CPU version test passed, search results: [[  0 940 487 737 164]]
   ```
   
   

5. run main.py



6. If wanna try web mode, 
```bash
pip install Flask==2.3.3 WTForms==3.1.1 Flask-WTF==1.2.1 Flask-Session==0.5.0
```
or
```bash
pip install -r requirements-web.txt
```

Then run:
```python
python app.py
```
and go to http://localhost:5000



7.We also provide a chunk analyze tool, if you would like to use it, do as:

Command:

```bash
python analyze_chunking.py --file <File path> [options]
```

Parameter Description

- `--file`, `-f`: Document path to be analyzed (required)
- `--model`, `-m`: Ollama model name, default is "mistral".
- `--chunk-sizes`: A list of chunk sizes to test, for example:`--chunk-sizes 500 1000 1500 2000`
- `--chunk-overlaps`: A list of chunked overlap sizes to test, for example:`--chunk-overlaps 50 100 200`
- `--query`, `-q`: Query statement for testing, defaults to "Please summarize the main points of this document".

Example:

```bash
# basic usage
python analyze_chunking.py --file "d:\NLP\QA\test.txt"

# Specify model and chunking parameters
python analyze_chunking.py --file "d:\NLP\QA\test.txt" --model "qwen" --chunk-sizes 300 600 900 --chunk-overlaps 30 60 90

# Customized Queries
python analyze_chunking.py --file "d:\NLP\QA\test.txt" --query "Who is the main character in this article?"
```

Use in your code:

You can also import and use the analysis tools in your own Python code:

```python
from Utils.chunking_analyzer import ChunkingAnalyzer

# Create an Analyzer Instance
analyzer = ChunkingAnalyzer(model_name="mistral")

# Run analysis
results = analyzer.analyze_chunking_strategy(
    file_path="path/to/document.txt",
    chunk_sizes=[500, 1000, 1500],
    chunk_overlaps=[50, 100, 200]
)

# Export results
analyzer.export_results()

# Visualize results
analyzer.visualize_results()
```

## 
