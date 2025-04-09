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
    """Security Detection Text File Encoding"""
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
        logging.error(f"Coding detection fail: {str(e)}")
        return 'utf-8'

def load_document(qa_system, file_path):
    """Document Loading Process"""
    print(f"📄  Read file：{file_path}")
    try:
        if not os.path.isfile(file_path):
            raise ValueError("The path is not a file.")

        # Code detection and alternative code lists
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
                    loader.load()  # Test loading
                    break
                except UnicodeDecodeError:
                    continue

        documents = loader.load()

        # Slicing and Dicing Documents with Recursive Text Splitters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "……", "…", "　"]
        )
        docs = text_splitter.split_documents(documents)

        print(f"✅ Successfully loaded {len(docs)} blocks of text.")
        if docs:
            print(f"📝 Example of a first text block:{docs[0].page_content[:200]}...")

        # Constructing a Vector Database
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 32}
        )

        qa_system.vector_db = FAISS.from_documents(
            docs,
            embeddings
        )

        # Initializing the Q&A Chain
        qa_system.qa_chain = RetrievalQA.from_chain_type(
            llm=qa_system.llm_registry[qa_system.current_model],
            chain_type="stuff",
            retriever=qa_system.vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return True
    except PermissionError as pe:
        logging.error(f"❌ Privilege denial: {str(pe)}")
        return False
    except Exception as e:
        logging.error(f"❌ Document Loading Exception. {str(e)}")
        return False
