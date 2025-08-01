from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class RAGPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.llm = OpenAI()

    def process_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Process documents and return chunks with metadata"""
        chunks = []
        for doc in documents:
            splits = self.text_splitter.split_text(doc)
            for i, split in enumerate(splits):
                chunks.append({
                    "text": split,
                    "metadata": {
                        "source": f"doc_{i}",
                        "chunk_index": i
                    }
                })
        return chunks

    def create_vector_store(self, chunks: List[Dict[str, Any]], collection_name: str):
        """Create vector store from document chunks"""
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        vector_store = Qdrant(
            collection_name=collection_name,
            embeddings=self.embeddings
        )
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        return vector_store

    def create_qa_chain(self, vector_store):
        """Create QA chain for document retrieval and response generation"""
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever
        )
        return qa_chain
