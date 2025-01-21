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
            schema = CollectionSchema(fields=cschema,
                                    description="Metadata and embedding for weburl and youtube videos!")
            self.db = Collection(name=collection_name, schema=schema)       
            print(f"Collection '{collection_name}' created successfully.")
            self.db.create_index(field_name="embeddings", index_params=index_params)
            print(f" Index for Collection '{collection_name}' created successfully.")
            
        except MilvusException as e:
            print(f"Failed to create collection '{collection_name}': {e}",e)
            raise

    def insert(self, collection_name: str, entities: dict):
        try: 
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

    def get_all_documents(self, collection_name: str):
        try:
            self.db.load()

            if self.db.is_empty:
                print(f"No documents found in the collection '{collection_name}'.")
                return None
            
            output_fields = [field.name for field in self.db.schema.fields]
            #output_fields=['description']
            print(output_fields)
            results = self.db.query(expr="", output_fields=output_fields, limit=10)
            return results
        except MilvusException as e:
            print(f"Failed to retrieve documents from collection '{collection_name}': {e}")
            raise

    def describe_collection(self, collection_name: str):
        try:
            collection = Collection(name=collection_name)
            schema = collection.schema
            indexes = collection.indexes
            indexed_fields = [index.field_name for index in indexes]
            print(f"Schema for collection '{collection_name}': {schema}")
            print(f"Indexes for collection '{collection_name}': {indexes}")
            print(f"Indexed fields for collection '{collection_name}': {indexed_fields}")
            return {"schema": schema, "indexes": indexes, "indexed_fields": indexed_fields}
        except MilvusException as e:
            print(f"Failed to describe collection '{collection_name}': {e}")
            raise

    def search_by_description(self, collection_name: str, description: str, embeddings_model, k: int = 10):
        try:
            self.db.load()
            # Embed the query description
            query_embedding = embeddings_model.encode([description])[0]
            if isinstance(query_embedding, str):
                query_embedding = json.loads(query_embedding)
            
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            results = self.db.search([query_embedding], "embeddings",
                                     search_params, limit=k, 
                                     output_fields=["view_count"])
            return results
        except MilvusException as e:
            print(f"Failed to search documents in collection '{collection_name}' with description '{description}': {e}")
            raise

    def search_by_view_count(self, collection_name: str, min_view_count: int, k: int = 10):
        try:
            self.db.load()
            expr = f"view_count > {min_view_count}"
            output_fields = [field.name for field in self.db.schema.fields]
            results = self.db.query(expr=expr, output_fields=output_fields, limit=k)
            return results
        except MilvusException as e:
            print(f"Failed to search documents in collection '{collection_name}' with comment count greater than {min_view_count}: {e}")
            raise

    def search_by_description_and_view_count(self, collection_name: str, description: str, embeddings_model, min_view_count: int, k: int = 10):
        try:
            self.db.load()
            # Embed the query description
            query_embedding = embeddings_model.encode([description])[0]
            print("Query Embedding is ", query_embedding)
            if isinstance(query_embedding, str):
                query_embedding = json.loads(query_embedding)
            
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 30},
            }
            expr = f"view_count > {min_view_count}"
            results = self.db.search([query_embedding], "embeddings",
                                     search_params, limit=k, 
                                     output_fields=["description", "view_count"], expr=expr)
            return results
        except MilvusException as e:
            print(f"Failed to search documents in collection '{collection_name}' with description '{description}' and comment count greater than {min_view_count}: {e}")
            raise

    def explain_search_by_description_and_comment_count(self, collection_name: str, description: str, embeddings_model, min_view_count: int, k: int = 10):
        try:
            self.db.load()
            # Embed the query description
            query_embedding = embeddings_model.encode([description])[0]
            print("Query Embedding:", query_embedding)  # Debug print
            if isinstance(query_embedding, str):
                query_embedding = json.loads(query_embedding)
            
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 30},
            }
            expr = f"comment_count > {min_view_count}"
            print("Search Params:", search_params)  # Debug print
            print("Expression:", expr)  # Debug print
            
            # Print explain plan
            explain_plan = self.db.explain([query_embedding], "embeddings",
                                           search_params, limit=k, 
                                           output_fields=["description", "comment_count"], expr=expr)
            print("Explain Plan:", explain_plan)  # Debug print
            return explain_plan
        except MilvusException as e:
            print(f"Failed to explain search plan for collection '{collection_name}'
                   with description '{description}' and comment count greater than {min_view_count}: {e}")
            raise
