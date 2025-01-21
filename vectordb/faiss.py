from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from .base import MYVectorDB

class FAISSVectorDB(MYVectorDB):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.db = None

    def create_collection(self, collection_name: str, schema: dict):
        # FAISS does not support collections, so this method is not applicable
        pass

    def insert(self, collection_name: str, documents: List[Document]):
        self.db = FAISS.from_documents(documents, self.embeddings)

    def query(self, collection_name: str, query: str, k: int):
        return self.db.similarity_search(query, k)

    def create_index(self, collection_name: str, index_params: dict):
        # FAISS automatically creates an index, so this method is not applicable
        pass

    def load(self, collection_name: str):
        # FAISS loads the database from a local file
        self.db = FAISS.load_local(collection_name, self.embeddings, allow_dangerous_deserialization=True)

    def delete(self, collection_name: str, expr: str):
        # FAISS does not support deletion, so this method is not applicable
        pass

    def drop_collection(self, collection_name: str):
        # FAISS does not support collections, so this method is not applicable
        pass

    def save(self, dbname: str):
        self.db.save_local(dbname)
