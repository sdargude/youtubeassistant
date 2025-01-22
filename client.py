import os
import json
import sys
sys.path.append('/Users/sdargude/playground/code/llms/youtubeassistant')
from vectordb import MilvusVectorDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusException
)
import ast

from sentence_transformers import SentenceTransformer

load_dotenv()

def read_transcript(transcript_path: str):
    with open(transcript_path, "r") as file:
        transcript = file.read()
    return transcript

def read_metadata(metadata_path: str):
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    return metadata

def create_document_from_metadata(metadata: dict, embeddings_model, expected_dim):
    # Generate embedding for the description
    embedding = embeddings_model.encode([metadata['description']])[0]
    if isinstance(embedding, str):
        embedding = json.loads(embedding)
    if len(embedding) != expected_dim:
        raise ValueError(f"Embedding dimension {len(embedding)} does not match expected dimension {expected_dim}")
    
    documents = []
    # Create a document with the embedding as content and the rest of the metadata
    documents.append(Document(page_content=metadata['description'], metadata=metadata))
    return documents

def split_text_into_documents(text: str, metadata: dict):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20, length_function=len, is_separator_regex=False)
    split_docs = text_splitter.split_documents([Document(page_content=text, metadata=metadata)])
    return split_docs

def generate_embeddings(documents, embeddings_model, expected_dim, text_field_name):
    for doc in documents:
        # Store the original text in the specified field of the metadata
        doc.metadata[text_field_name] = doc.page_content
        
        # Generate embeddings for the document content
        embedding = embeddings_model.encode([doc.page_content])[0]
        # Ensure the embeddings are in the correct format (list of floats) and dimension
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        if len(embedding) != expected_dim:
            raise ValueError(f"Embedding dimension {len(embedding)} does not match expected dimension {expected_dim}")
        doc.page_content = embedding
    return documents

def create_metadata_collection(milvus_db: MilvusVectorDB, collection_name:str):

    dim = 384  # Update dimension to match all-MiniLM-L6-v2
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="uri", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=250),
        FieldSchema(name="description", dtype=DataType.VARCHAR,max_length=5024),
        FieldSchema(name="publish_date", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="view_count", dtype=DataType.INT64 ),   
        FieldSchema(name="comment_count", dtype=DataType.INT64),  # Change type to INT64
         FieldSchema(name="like_count", dtype=DataType.INT64 ), 
        FieldSchema(name="dislike_count", dtype=DataType.INT64 ), 
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    # Create collection
    milvus_db.create_collection(collection_name, fields)

def create_transcript_collection(milvus_db: MilvusVectorDB, collection_name: str):
    dim = 384  # Update dimension to match all-MiniLM-L6-v2
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    # Create collection
    milvus_db.create_collection(collection_name, fields)
 

def construct_metadata_entities(documents):
    entities = {
        "uri": [],
        "source_type": [],
        "title": [],
        "description": [],
        "publish_date": [],
        "view_count": [],
        "dislike_count": [],
        "like_count": [],
        "comment_count": [],
        "embeddings": []
    }
    
    for doc in documents:
        uri = doc.metadata['id']
        source_type = doc.metadata.get('source_type', '')
        title = doc.metadata.get('title', '')
        description = doc.metadata.get('description', '')
        publish_date = doc.metadata.get('publish_date', '')
        view_count = int(doc.metadata.get('view_count', 0))  # Convert to int
        dislike_count = int(doc.metadata.get('dislike_count', 0))  # Convert to int
        like_count = int(doc.metadata.get('like_count', 0)) 
        comment_count = int(doc.metadata.get('comment_count', 0))  # Convert to int
        embeddings = doc.page_content
         
        entities["uri"].append(uri)
        entities["source_type"].append(source_type)
        entities["title"].append(title)
        entities["description"].append(description)
        entities["publish_date"].append(publish_date)
        entities["view_count"].append(view_count)
        entities["dislike_count"].append(dislike_count)
        entities["like_count"].append(dislike_count)
        entities["comment_count"].append(comment_count)
        entities["embeddings"].append(embeddings)

    #print("Entities:", entities)  # Print the entire entities dictionary

    return entities


def process_metadata_file(metadata_path, embeddings, expected_dim, milvus_db, collection_name):
    print("In process_metadata :", metadata_path)
    metadata = read_metadata(metadata_path)
    
    description = metadata['description']
    # Copy everything but the description field into another dictionary
    metadata_copy = {k: v for k, v in metadata.items() if k != 'description'}
    
    # Split text into documents
    metadata_documents = split_text_into_documents(description, metadata_copy)
    
    # Generate embeddings for documents
    meta_embedded_documents = generate_embeddings(metadata_documents, embeddings, expected_dim, text_field_name="description")

    # Construct entities for insertion
    entities = construct_metadata_entities(meta_embedded_documents)
    
    # Insert documents into Milvus
    milvus_db.insert(collection_name, entities)
    print(f"Inserted documents from {metadata_path} into collection '{collection_name}'")

def process_transcript_file(transcript_path, metadata, embeddings, expected_dim, milvus_db, collection_name):
    print("In process_transcript:", transcript_path)
    transcript = read_transcript(transcript_path)
    
    # Split text into documents with id as metadata
    metadata_copy = {"id": metadata["id"]}
    transcript_documents = split_text_into_documents(transcript, metadata_copy)
    
    # Generate embeddings for documents
    transcript_embedded_documents = generate_embeddings(transcript_documents, embeddings, expected_dim, text_field_name="text")

    # Construct entities for insertion
    entities = {
        "id": [],
        "text": [],
        "embeddings": []
    }
    
    for doc in transcript_embedded_documents:
        entities["id"].append(doc.metadata["id"])
        entities["text"].append(doc.metadata["text"])
        entities["embeddings"].append(doc.page_content)
    
    print("Number of transcript embedded documents...", len(transcript_embedded_documents))
    # Insert documents into Milvus
    milvus_db.insert(collection_name, entities)
    print(f"Inserted documents from {transcript_path} into collection '{collection_name}'")

def create_and_insert_data(transcript_dir, embeddings, expected_dim, milvus_db):
    # Create collections
    create_metadata_collection(milvus_db, collection_name="transcript_metadata")
    create_transcript_collection(milvus_db, collection_name="transcript_collection")
    
    # Read transcript and metadata files
    for filename in os.listdir(transcript_dir):
        if filename.startswith("META_"):
            print("Skipping...", filename)
            continue
        
        transcript_path = os.path.join(transcript_dir, filename)
        metadata_path = os.path.join(transcript_dir, f"META_{filename.replace('.txt', '.json')}")
        
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found for {filename}")
            continue
        
        print("Now processing.....", metadata_path)
        metadata = read_metadata(metadata_path)
        process_metadata_file(metadata_path, embeddings, expected_dim, milvus_db, collection_name="transcript_metadata")
        process_transcript_file(transcript_path, metadata, embeddings, expected_dim, milvus_db, collection_name="transcript_collection")
        
    print("Collections.....")
    print(milvus_db.list_collections())

def query_data(milvus_db, embeddings):
    # Retrieve and print all documents from the collection with an optional filter condition and output fields
    filter_condition = 'id == "BBBBBBBBBBBBBBBBBBBB"'
    output_fields = ['text']
    all_documents = milvus_db.get_all_documents(collection_name="transcript_collection",
                                                 filter_condition=filter_condition,
                                                 output_fields=output_fields)
    print("Filtered documents in 'transcript_collection':", len(all_documents))
 
    return
    # Search for documents with a specific description
    search_results = milvus_db.search_by_description(
        collection_name="transcript_metadata", 
        description="red fruit", 
        embeddings_model=embeddings
    )
    #for result in search_results:
    #    print("Search results for description 'red fruit':", result) 
    # Search for documents with comment count greater than 1000
    comment_count_results = milvus_db.search_by_view_count(
        collection_name="transcript_metadata", 
        min_view_count=1000
    ) 
    #print("Here Here Search results for comment count greater than 1000:", comment_count_results)
     
    # Search for documents with description similarity and comment count greater than 1000
    combined_results = milvus_db.search_by_description_and_view_count(
        collection_name="transcript_metadata", 
        description="red fruit", 
        embeddings_model=embeddings, 
        min_view_count=1000
    )
    for result in combined_results:
        print("Search results for description 'red fruit' and comment count greater than 1000:", result)

def main():
    transcript_dir = "transcripts"
    expected_dim = 384  # Dimension for all-MiniLM-L6-v2
    
    # Initialize embeddings
    embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize MilvusVectorDB
    milvus_db = MilvusVectorDB(embeddings)
    
    # Create and insert data
    #create_and_insert_data(transcript_dir, embeddings, expected_dim, milvus_db)
    
    # Query data
    query_data(milvus_db, embeddings)

if __name__ == "__main__":
    main()
