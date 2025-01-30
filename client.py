import os
import sys
import json
sys.path.append('/Users/sdargude/playground/code/llms/youtubeassistant')
from vectordb import MilvusVectorDB
from metadata import create_metadata_collection, process_metadata_file, search_metadata  # Import the function
from transcript import create_transcript_collection, process_transcript_file
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv 
from langchain.schema import Document  # Import LangChain Document class
from utils import read_metadata  # Import the function from utils

load_dotenv()

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

def parent_retriever(search_results):
    documents = []
    for result in search_results:
        transcript_path = result.get('transcript_path')
        if not transcript_path:
            print("Transcript path not found in result.")
            return []
        
        start_offset = result['start']
        end_offset = result['end']
         
        with open(transcript_path, 'r') as file:
            transcript_text = file.read()
            extracted_text = transcript_text[start_offset:end_offset]
            # Remove embeddings field from result
            result.pop('embeddings', None)
            documents.append(Document(page_content=extracted_text, metadata=result))
    
    return documents

def query_data(milvus_db, embeddings):
    # Retrieve and print all documents from the collection with an optional filter condition and output fields
    filter_condition = 'id == "Xv5nBumG2sw"'
    #filter_condition = 'id == "BBBBBBBBBBBBBBBBBBBB"'
    output_fields = ['text']
    """
    try:
        all_documents = milvus_db.get_all_documents(collection_name="transcript_collection",
                                                 filter_condition=filter_condition,
                                                 output_fields=None)
        print("Filtered documents in 'transcript_collection':", len(all_documents))
    
        # Use parent_retriever to extract text
        langchain_documents = parent_retriever(all_documents)
        print("LangChain documents:", langchain_documents)
    except ValueError as e:
        print("No data...",e)
    """

    #print(all_documents[0])
    # Search for documents with a specific description
    #search_results = milvus_db.search_by_description(
    #    collection_name="transcript_metadata", 
    #    description="red fruit", 
    #    embeddings_model=embeddings
    #)
    #for result in search_results:
    #    print("Search results for description 'red fruit':", result) 
    # Search for documents with comment count greater than 1000
    #comment_count_results = milvus_db.search_by_view_count(
    #    collection_name="transcript_metadata", 
    #    min_view_count=1000
    #) 
    #print("Here Here Search results for comment count greater than 1000:", comment_count_results)
     
    # Search for documents with description similarity and comment count greater than 1000
    
    combined_results = search_metadata(
        milvus_db=milvus_db, 
        collection_name="transcript_metadata", 
        #data="How to construct a pest-proof garden bed box to protect plants from moles, \
        #    raccoons, skunks, birds, and insects. Including methods for soil leveling,\
        #      hardware cloth, and frame building.",
        data=" STOCK Microsoft analysis",
        embeddings_model=embeddings, 
        topk=10, 
        output_fields=["title", "id", "embeddings"], 
        expr="",
        text=True,  # Set text to True to get the actual text instead of embeddings
    )
 
    if len(combined_results) == 0:
        print("Search metadata has no results.....")
    else:
        print("Number of document matched after metadata search ", len(combined_results))
    
        for result in combined_results:
            if result.metadata['distance'] < 0.7:
                print("Search results for metadata", result.metadata)
                print("==========")

def main():
    transcript_dir = "transcripts"
    expected_dim = 384  # Dimension for all-MiniLM-L6-v2
    
    # Initialize embeddings
    embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize MilvusVectorDB
    milvus_db = MilvusVectorDB(embeddings)

    #milvus_db.drop_collection("transcript_collection")
    #milvus_db.drop_collection("youtube_weburl_collection")
    #milvus_db.drop_collection("transcript_metadata")
    
    
    # Create and insert data
    #create_and_insert_data(transcript_dir, embeddings, expected_dim, milvus_db)
    
    #milvus_db.describe_collection("transcript_metadata")
    # Query data
    query_data(milvus_db, embeddings)

if __name__ == "__main__":
    main()
