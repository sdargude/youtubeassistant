from youtube import Youtube
from weburl import WebURL
from TranscriptFactory import TranscriptFactory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.youtube import TranscriptFormat

from langchain_ollama import OllamaLLM
from langchain_openai import OpenAI
from langchain_ollama import OllamaEmbeddings

import os
import hashlib
from typing import List, Type
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
embeddings = OllamaEmbeddings(model='llama3')
small_model = "llama3.1"
big_model = "llama3.1:70b"

class VectorDBFactory:
    @staticmethod
    def create_vector_db(db_type: str, documents: List[Document], dbname: str):
        if db_type == "FAISS":
            db = FAISS.from_documents(documents, embeddings)
            db.save_local(dbname)
            return db
        # Add other database types here
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

def get_transcript_file_path(url: str) -> str:
    transcript_instance = TranscriptFactory.create_transcript(url)
    if isinstance(transcript_instance, Youtube):
        video_id = transcript_instance.extract_video_id(url)
        transcript_path = os.path.join("transcripts", f"{transcript_instance.title}.txt")
    else:
        video_id = transcript_instance.url.split("/")[-1]
        transcript_path = os.path.join("transcripts", f"{transcript_instance.title}.txt")
    return transcript_path

def create_vector_db_from_transcript_file(transcript_path: str, dbname: str, db_type: str = "FAISS") -> Type:
    vdb_path = os.path.join("vdb", dbname)
    print("DBNAME : ", vdb_path)
    if os.path.exists(vdb_path):
        if db_type == "FAISS":
            print("Loading the existing db from:", vdb_path)
            newdb = FAISS.load_local(vdb_path, embeddings, allow_dangerous_deserialization=True)
            return newdb
        # Add other database types here
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    with open(transcript_path, "r") as file:
        transcript = file.read()

    textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, length_function=len, is_separator_regex=False)
    all_split_docs = textsplitter.split_documents([Document(page_content=transcript)])
    
    db = VectorDBFactory.create_vector_db(db_type, all_split_docs, vdb_path)
    print(f"Vector database saved to: {vdb_path}")
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k)
    print("Retrieved", len(docs), "Documents")
    docspagecontent = " ".join([d.page_content for d in docs])

    llm = OllamaLLM(model=small_model)

    prompt = PromptTemplate(
        input_variables=['question', "docs"],
        template="""
        you are a helpful Youtube assistant that can answer questions about videos based on the video's transcript. 
        Answer the following question {question} by searching the following video transcript: {docs}
        Only use the factual information from the transcript to answer the question. 
        If you feel like you don't have enough information to answer the question say "I don't know". 
        your answer should be one paragraph, Email has format "#####@###.###". Output format should be
        Title: <Text> <newline>
        Summary:
        """
    )
    chain = prompt | llm
    response = chain.invoke({"question": query, "docs": docspagecontent})
    response = response.replace("\n", "")
    return response

def main():
    url = "https://www.youtube.com/watch?v=zm0QVutAkYg"  # Replace with the actual URL
    
    # Get the transcript file path
    transcript_path = get_transcript_file_path(url)
    
    # Create transcript if it does not exist
    if not os.path.exists(transcript_path):
        transcript_instance = TranscriptFactory.create_transcript(url)
        transcript_instance.save_transcript_to_file(transcript_path)
    
    # Create vector database
    db_type = "FAISS"  # Specify the database type
    if db_type == "FAISS":
        with open(transcript_path, "rb") as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        dbname = checksum
    else:
        dbname = "example_vdb"
    
    db = create_vector_db_from_transcript_file(transcript_path, dbname, db_type)
    
    # Get response from query
    query = "What is the video about?"
    response = get_response_from_query(db, query)
    print(response)

if __name__ == '__main__':
    main()

