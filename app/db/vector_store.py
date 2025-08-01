from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

class VectorStoreManager:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, collection_name: str, vector_size: int):
        """Create a new vector collection"""
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    def add_documents(self, collection_name: str, documents: List[Dict[str, any]]):
        """Add documents to vector store"""
        points = []
        for idx, doc in enumerate(documents):
            points.append({
                "id": idx,
                "vector": doc["embedding"],
                "payload": doc["metadata"]
            })
        
        self.client.upsert(collection_name=collection_name, points=points)

    def search(self, collection_name: str, query_vector: List[float], limit: int = 5):
        """Search for similar documents"""
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
