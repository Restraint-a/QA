# document_loader.py
import os
import logging
import torch
import chardet
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

device = "cuda" if torch.cuda.is_available() else "cpu"
def detect_encoding(file_path, max_size=1024*1024):
    """安全检测文本文件编码"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = b''
            chunk_size = 4096
            while len(raw_data) < max_size:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                raw_data += chunk
            return chardet.detect(raw_data)['encoding']
    except Exception as e:
        logging.error(f"编码检测失败: {str(e)}")
        return 'utf-8'

def load_document(qa_system, file_path):
    """文档加载处理"""
    print(f"📄  读取文件：{file_path}")
    try:
        if not os.path.isfile(file_path):
            raise ValueError("路径不是文件")

        # 编码检测和备选编码列表
        encoding = detect_encoding(file_path)
        retry_encodings = [encoding, 'utf-8', 'gbk']

        loader = None
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(file_path)
        else:
            for enc in retry_encodings:
                try:
                    loader = TextLoader(file_path, encoding=enc)
                    loader.load()  # 测试加载
                    break
                except UnicodeDecodeError:
                    continue

        documents = loader.load()

        # 使用递归文本分割器对文档进行切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "；", "……", "…", "　"]
        )
        docs = text_splitter.split_documents(documents)

        print(f"✅ 成功加载 {len(docs)} 个文本块")
        if docs:
            print(f"📝 首文本块示例：{docs[0].page_content[:200]}...")

        # 构造向量数据库
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 32}
        )

        qa_system.vector_db = FAISS.from_documents(
            docs,
            embeddings
        )

        # 初始化问答链
        qa_system.qa_chain = RetrievalQA.from_chain_type(
            llm=qa_system.llm_registry[qa_system.current_model],
            chain_type="stuff",
            retriever=qa_system.vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return True
    except PermissionError as pe:
        logging.error(f"权限拒绝: {str(pe)}")
        return False
    except Exception as e:
        logging.error(f"文档加载异常: {str(e)}")
        return False
