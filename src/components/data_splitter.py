from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class DataSplitter:
    def __init__(self,text_splitter: RecursiveCharacterTextSplitter):
        self.text_splitter = text_splitter

    def split_data(self,data: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(data)



