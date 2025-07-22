from typing import Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def load_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Load embedding model"""
        if model_name not in self.models:
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            self.models[model_name] = AutoModel.from_pretrained(model_name)
        return self.models[model_name]

    def get_embeddings(self, text: str, model_name: str) -> torch.Tensor:
        """Get embeddings for text"""
        model = self.load_embedding_model(model_name)
        tokenizer = self.tokenizers[model_name]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def load_llm(self, model_name: str = "gpt-3.5-turbo"):
        """Load LLM model"""
        # TODO: Implement LLM loading logic
        return None
