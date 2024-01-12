import streamlit as st
import langchainhelper as lch
import textwrap
import os
 
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
