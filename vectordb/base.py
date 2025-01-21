from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class MYVectorDB(ABC):
    @abstractmethod
    def create_collection(self, collection_name: str, schema: dict):
        pass

    @abstractmethod
    def insert(self, collection_name: str, documents: List[Document]):
        pass

    @abstractmethod
    def query(self, collection_name: str, query: str, k: int):
        pass

   
    @abstractmethod
    def load(self, collection_name: str):
        pass

    @abstractmethod
    def delete(self, collection_name: str, expr: str):
        pass

    @abstractmethod
    def drop_collection(self, collection_name: str):
        pass
