import os
import time
import torch
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
                num_ctx=4096
            ),
            "qwen": Ollama(
                model="qwen:1.8b",
                temperature=0.5,
                num_ctx=2048
            )
        }

    def _release_model_resources(self):
        """彻底释放模型资源"""
        for model in self.llm_registry.values():
            if hasattr(model, 'client'):
                del model.client
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
