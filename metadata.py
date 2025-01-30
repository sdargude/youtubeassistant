import os
import ast
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import FieldSchema, DataType
from sentence_transformers import SentenceTransformer
from vectordb import MilvusVectorDB

def read_metadata(metadata_path: str):
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    return metadata

def create_metadata_collection(milvus_db: MilvusVectorDB, collection_name: str):
    dim = 384  # Update dimension to match all-MiniLM-L6-v2
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=250),
        FieldSchema(name="publish_date", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="view_count", dtype=DataType.INT64),   
        FieldSchema(name="comment_count", dtype=DataType.INT64),  # Change type to INT64
        FieldSchema(name="like_count", dtype=DataType.INT64), 
        FieldSchema(name="dislike_count", dtype=DataType.INT64), 
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    # Create collection
    milvus_db.create_collection(collection_name, fields)

def construct_metadata_entities(documents):
    entities = {
        "id": [],
        "source_type": [],
        "title": [],
        "publish_date": [],
        "view_count": [],
        "dislike_count": [],
        "like_count": [],
        "comment_count": [],
        "embeddings": []
    }
    
    for doc in documents:
        id = doc.metadata['id']
        source_type = doc.metadata.get('source_type', '')
        title = doc.metadata.get('title', '')
        publish_date = doc.metadata.get('publish_date', '')
        view_count = int(doc.metadata.get('view_count', 0))  # Convert to int
        dislike_count = int(doc.metadata.get('dislike_count', 0))  # Convert to int
        like_count = int(doc.metadata.get('like_count', 0)) 
        comment_count = int(doc.metadata.get('comment_count', 0))  # Convert to int
        embeddings = doc.page_content
         
        entities["id"].append(id)
        entities["source_type"].append(source_type)
        entities["title"].append(title)
        entities["publish_date"].append(publish_date)
        entities["view_count"].append(view_count)
        entities["dislike_count"].append(dislike_count)
        entities["like_count"].append(like_count)
        entities["comment_count"].append(comment_count)
        entities["embeddings"].append(embeddings)

    return entities

def process_metadata_file(metadata_path, embeddings, expected_dim, milvus_db, collection_name):
    print("In process_metadata:", metadata_path)
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

def search_metadata(milvus_db, collection_name, data, 
                    embeddings_model, topk=10, output_fields=None, text=True, expr="" , params=None):
    
    description_embedding = embeddings_model.encode(data)
   
    # Build the query based on the provided conditions 
    if params == None:
        search_params = {"metric_type": "L2",
                        "params": {
                            "radius": 0.6,
                            "range_filter":1.0
                            }
                        }
    else:
        search_params = params

    if output_fields is None:
        output_fields = ["view_count"]

    if "embeddings" in output_fields and text:
        if "title" not in output_fields:
            output_fields.append("title")
     
    try:
        results = milvus_db.search(collection_name, 
                                   description_embedding=description_embedding, 
                                   params=search_params,
                                   expr=expr,
                                   output_fields=output_fields,
                                   topk=10)
    except ValueError as e:
        print(f"Failed to perform search on collection '{collection_name}': {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during search: {e}")
        return []
    
    documents = []
    #print(results)

    # If 'embeddings' is in output_fields and text is True, replace embeddings with actual text
    if "embeddings" in output_fields and text:
        for result in results:
            i=0
            for hit in result:
                metadata = {} 
                metadata['distance'] = results[0].distances[i]
                i+=1
      
                for field in output_fields:
                    if field == "embeddings":
                        # Construct the file name from the title field
                        title = hit.entity.get("title")
                   
                        if title:
                            metadata['title'] = title
                            metadata_file_path = os.path.join(os.path.dirname(__file__), 
                                                            "transcripts", f"META_{title}.json") 
                            if os.path.exists(metadata_file_path):
                                with open(metadata_file_path, "r") as file:
                                    metadata_content = json.load(file)
                                    description = metadata_content.get("description", "")
                                    metadata[field] = description
                            else:
                                metadata[field] = "Metadata file not found"
                        else:
                            metadata[field] = "Title not found"
                    else:
                        try:
                            metadata[field] = hit.entity.get(field)
                        except Exception as e:
                            print("Field ", field, "Not found in the metadata...")
                  

                description = None
                if "embeddings" in metadata:
                    description = metadata['embeddings'] 
                    del metadata['embeddings'] 

                documents.append(Document(page_content=description, metadata=metadata))
    
    return documents