from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os


load_dotenv()

embeddings = OpenAIEmbeddings()

def createVectorDBfromYoutubeUrl(videourl:str, dbname:str)->FAISS:

    if os.path.exists(dbname): 
        newdb = FAISS.load_local(dbname, embeddings)
        print("Loading the data from :", dbname)
        return newdb
    
    loader = YoutubeLoader.from_youtube_url(videourl)
    transcript = loader.load()
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    docs = textsplitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(dbname)
    return db


def getresponsefromquery(db,  query, k=4):
    #text-davinci can handle 4097 tokens
    docs = db.similarity_search(query,k)

    docspagecontent = " ".join([d.page_content for d in docs])

    #llm = OpenAI(model="text-davinci-003")

    llm = OpenAI(model="gpt-3.5-turbo-instruct")
    
    prompt = PromptTemplate(
        input_variables = ['question', "docs"] ,
        template = """
        you are a helpful Youtube assistant that can answer questions about videos based on the video's transcript. 
        Answer the following question {question} by searching the following video transcript : {docs}
        Only use the factual information from the transcript to answer the question. 
        If you feel like you don't have enough information to answer the question say "I don't know" 
        your answer should be maximum two sentences. Email has format "#####@###.###".
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run (question=query, docs = docspagecontent)
    response = response.replace("\n", "")
    return response


if __name__ == '__main__':
    #db=createVectorDBfromYoutubeUrl("https://www.youtube.com/watch?v=lG7Uxts9SXs&t=1895s", dbname="YT123")
    question = "who is satya nadella?"
    #response = getresponsefromquery (db, question)

    question = "who is Sharukh khan?"
    #response = getresponsefromquery (db, question)

    question = "get all the email address, email address has format string@string.string ?"
    response = getresponsefromquery (db, question)

    

    
    print(response)

 