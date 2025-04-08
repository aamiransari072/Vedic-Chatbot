from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List


class DataEmbeddings:
    def __init__(self,model_name: str,model_kwargs: dict):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
    
    def embed_data(self,data: List[Document]) -> List[Document]:
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs
        )
        return embeddings.embed_documents(data)

