# vectordb/__init__.py
from sys import exit
# Importing the necessary classes to expose them as part of the package API
try:
    from vectordb.base import MYVectorDB
    print("Imported VectorDB from base.py")
except ImportError as e:
    print("Failed to import vectorDB: {e}",e)

 
from .faiss import FAISSVectorDB
from .milvus import MilvusVectorDB

# Specifying what gets imported when someone does `from vectordb import *`
__all__ = ["VectorDB", "FAISSVectorDB", "MilvusVectorDB"]