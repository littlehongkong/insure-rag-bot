from typing import List, Dict
from PyPDF2 import PdfReader
from app.core.rag_pipeline import RAGPipeline

class PDFProcessor:
    def __init__(self):
        self.rag_pipeline = RAGPipeline()

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process PDF file and return chunks"""
        raw_text = self.extract_text(pdf_path)
        return self.rag_pipeline.process_documents([raw_text])

    def create_policy_index(self, pdf_path: str, collection_name: str):
        """Create vector store index for insurance policy"""
        chunks = self.process_pdf(pdf_path)
        return self.rag_pipeline.create_vector_store(chunks, collection_name)
