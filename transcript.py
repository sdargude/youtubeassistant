import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import FieldSchema, DataType
from sentence_transformers import SentenceTransformer
from vectordb import MilvusVectorDB

def read_transcript(transcript_path: str):
    with open(transcript_path, "r") as file:
        transcript = file.read()
    return transcript

def create_transcript_collection(milvus_db: MilvusVectorDB, collection_name: str):
    dim = 384  # Update dimension to match all-MiniLM-L6-v2
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="start", dtype=DataType.INT64),
        FieldSchema(name="end", dtype=DataType.INT64),
        FieldSchema(name="transcript_path", dtype=DataType.VARCHAR, max_length=255),  # Add transcript_path field
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    # Create collection
    milvus_db.create_collection(collection_name, fields)

def split_text_into_documents(text: str, metadata: dict):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20, length_function=len, is_separator_regex=False)
    split_docs = []
    start_offset = 0

    for chunk in text_splitter.split_text(text):
        end_offset = start_offset + len(chunk)
        chunk_metadata = metadata.copy()
        chunk_metadata["start"] = start_offset
        chunk_metadata["end"] = end_offset
        split_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
        start_offset = end_offset

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

def process_transcript_file(transcript_path, metadata, embeddings, expected_dim, milvus_db, collection_name):
    print("In process_transcript:", transcript_path)
    transcript = read_transcript(transcript_path)
    
    # Split text into documents with id, start, end, and transcript_path as metadata
    metadata_copy = {"id": metadata["id"], "transcript_path": transcript_path}
    transcript_documents = split_text_into_documents(transcript, metadata_copy)
    
    # Generate embeddings for documents
    transcript_embedded_documents = generate_embeddings(transcript_documents, embeddings, expected_dim, text_field_name="text")

    # Construct entities for insertion
    entities = {
        "id": [],
        "start": [],
        "end": [],
        "transcript_path": [],  # Add transcript_path field
        "embeddings": []
    }
    
    for doc in transcript_embedded_documents:
        entities["id"].append(doc.metadata["id"])
        entities["start"].append(doc.metadata["start"])
        entities["end"].append(doc.metadata["end"])
        entities["transcript_path"].append(doc.metadata["transcript_path"])  # Add transcript_path field
        entities["embeddings"].append(doc.page_content)
    
    print("Number of transcript embedded documents...", len(transcript_embedded_documents))
    # Insert documents into Milvus
    milvus_db.insert(collection_name, entities)
    print(f"Inserted documents from {transcript_path} into collection '{collection_name}'")
