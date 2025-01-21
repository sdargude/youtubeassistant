import os
import json
import sys
sys.path.append('/Users/sdargude/playground/code/llms/youtubeassistant')
from vectordb import MilvusVectorDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusException
)


from langchain_ollama import OllamaEmbeddings

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
    embedding = embeddings_model.embed_documents([metadata['description']])[0]
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
    print("Len of the split docs ", len(split_docs))
    print(split_docs)
    return split_docs

def generate_embeddings(documents, embeddings_model, expected_dim):
    for doc in documents:
        # Store the original text in the description field of the metadata
        doc.metadata['description'] = doc.page_content
        
        # Generate embeddings for the document content
        embedding = embeddings_model.embed_documents([doc.page_content])[0]
        # Ensure the embeddings are in the correct format (list of floats) and dimension
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        if len(embedding) != expected_dim:
            raise ValueError(f"Embedding dimension {len(embedding)} does not match expected dimension {expected_dim}")
        doc.page_content = embedding
    return documents


def create_metadata_collection(milvus_db: MilvusVectorDB, collection_name:str):

    index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
                }
    dim=4096
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="uri", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=250),
        FieldSchema(name="description", dtype=DataType.VARCHAR,max_length=5024),
        FieldSchema(name="publish_date", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="view_count", dtype=DataType.VARCHAR, max_length=20),   
        FieldSchema(name="comment_count", dtype=DataType.VARCHAR, max_length=20), 
        FieldSchema(name="dislike_count", dtype=DataType.VARCHAR, max_length=20), 
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    collection_name = "transcript_metadata"
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
        "comment_count": [],
        "embeddings": []
    }
    
    for doc in documents:
        entities["uri"].append(doc.metadata['id'])
        entities["source_type"].append(doc.metadata.get('source_type', ''))
        entities["title"].append(doc.metadata.get('title', ''))
        entities["description"].append(doc.metadata.get('description', ''))  # Store the text content in the description field
        entities["publish_date"].append(doc.metadata.get('publish_date', ''))
        entities["view_count"].append(doc.metadata.get('view_count', "0"))
        entities["dislike_count"].append(doc.metadata.get('dislike_count', "0"))
        entities["comment_count"].append(doc.metadata.get('comment_count', "0"))
        entities["embeddings"].append(doc.page_content)

    
    print("----------")
    print(entities)
    print("++++++++++++++++++++++++")
    
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
    meta_embedded_documents = generate_embeddings(metadata_documents, embeddings, expected_dim)

    print(meta_embedded_documents)
 
    # Construct entities for insertion
    entities = construct_metadata_entities(meta_embedded_documents)
    
    # Insert documents into Milvus
    milvus_db.insert(collection_name, entities)
    print(f"Inserted documents from {metadata_path} into collection '{collection_name}'")


def main():
    transcript_dir = "transcripts"
    schema_file = os.path.join("schema", "milvus_schema.json")
    collection_name = "youtube_weburl_collection"
    expected_dim = 4096
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model='llama3')
    
    # Initialize MilvusVectorDB
    milvus_db = MilvusVectorDB(embeddings)
    
    # Create collection
    #milvus_db.create_collection(collection_name, schema_file)
    create_metadata_collection(milvus_db, collection_name="transcript_metadata")
    
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
        process_metadata_file(metadata_path, embeddings, expected_dim, milvus_db, collection_name="transcript_metadata")
        
        print("Collections.....")
        milvus_db.get_client(url="http://localhost:19530", token="root:Milvus")
        print(milvus_db.list_collections())
        break

if __name__ == "__main__":
    main()
