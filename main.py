import streamlit as st
import langchainhelper as lch
import textwrap
import os
from youtube import Youtube
from weburl import WebURL

def extract_video_id(youtube_url):
    # Extract the video ID from the YouTube URL
    video_id = youtube_url.split("v=")[1]
    return video_id


def create_or_load_db(youtubeurl):
    db_directory = "vdb"
    os.makedirs(db_directory, exist_ok=True)

    video_id = extract_video_id(youtubeurl)
    db_filename = os.path.join(db_directory, f"{video_id}.db")
     
    
    db = lch.createVectorDBfromYoutubeUrl(youtubeurl, db_filename)
    return db
    
st.title("Youtube Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        youtubeurl = st.text_area(
            label="What is the Youtube Video URL?",
            max_chars=100,
        )

        query = st.text_area(
            label="Ask me about the video",
            key="query",
            max_chars=100
        )

        st.form_submit_button(label="Submit")
      


if query and youtubeurl: 
    db = create_or_load_db(youtubeurl)
    response = lch.getresponsefromquery (db, query)
    st.subheader("Answer: ")
    st.text(textwrap.fill(response, width=80))

def main():
    api_key = os.getenv("YOUTUBE_API_KEY")
    youtube_url = "https://www.youtube.com/watch?v=zm0QVutAkYg"
    
    yt = Youtube(api_key, youtube_url)
    
    # Download and save transcript
    documents = yt.download_youtube_transcript()
    if documents:
        yt.save_documents_to_file(documents)
    
    # Save transcript with title
    file_path, file_size = yt.save_transcript_with_title()
    if file_path:
        print(f"Transcript saved at {file_path} with size {file_size} bytes")

    # Example usage of WebURL
    web_url = "https://example.com"
    web = WebURL(web_url)
    web.save_transcript_to_file()

if __name__ == "__main__":
    main()
