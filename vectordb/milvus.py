import json
import os
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Milvus
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusException
)
from dotenv import load_dotenv
from vectordb.base import MYVectorDB
from pymilvus import DataType, CollectionSchema, Collection, MilvusClient

load_dotenv()

class MilvusVectorDB(MYVectorDB):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.db = None
        self.client = None
        self.connect()


    def get_client(self,url, token):
        try:
            self.client = MilvusClient(
            uri=url,
            token=token
            )
            return self.client
        except MilvusException as e:
            print(f"Failed to create client to Milvus: {e}")
            raise

    def connect(self):
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        profile = os.getenv("MILVUS_PROFILE", "default")
        try:
            connections.connect(profile, host=host, port=port)
            print(f"Connected to Milvus at {host}:{port} with profile {profile}")
        except MilvusException as e:
            print(f"Failed to connect to Milvus: {e}")
            raise
 

    def create_collection(self, collection_name: str, cschema):
        if utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' already exists. Deleting it.")
            utility.drop_collection(collection_name)
        
        print("Creating new Collection!!!") 
        
        index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
                } 
        try:
            schema = CollectionSchema(fields=cschema, index_params={"embedding": index_params},
                                    description="Metadata and embedding for weburl and youtube videos!")
            self.db = Collection(name=collection_name, schema=schema)       
            print(f"Collection '{collection_name}' created successfully.", self.db)
            
        except MilvusException as e:
            print(f"Failed to create collection '{collection_name}': {e}",e)
            raise

    def insert(self, collection_name: str, entities: dict):
        try:
            print("Trying to insert ..", self.db)
            self.db.insert([entities[field] for field in entities])
            self.db.flush()
            print(f"Inserted documents into collection '{collection_name}'.")
        except MilvusException as e:
            print(f"Failed to insert documents into collection '{collection_name}': {e}")
            raise

    def query(self, collection_name: str, query: str, k: int):
        self.db.load()
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        return self.db.search(query, "embeddings", search_params, limit=k)
 
    def load(self, collection_name: str):
        self.db.load()

    def delete(self, collection_name: str, expr: str):
        self.db.delete(expr)
 
    def list_collections(self):
        if self.client:
            res =self.client.list_collections()
            print("List is ",res)
            return res
        return []

    def drop_collection(self, collection_name: str):
        utility.drop_collection(collection_name)
