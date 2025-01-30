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

def main():
    api_key = os.getenv("YOUTUBE_API_KEY")
    #youtube_url = "https://www.youtube.com/watch?v=zm0QVutAkYg"
    youtube_url = "https://www.youtube.com/watch?v=Xv5nBumG2sw"
    youtube_url = "https://www.youtube.com/watch?v=-Db9RJfze0g"
    
    yt = Youtube(api_key, youtube_url)

    # Save transcript with title
    file_path, file_size = yt.save_transcript_to_file()
    if file_path:
        print(f"Transcript saved at {file_path} with size {file_size} bytes")
  
    # Example usage of WebURL
    web_url = "https://finance.yahoo.com/quote/ASML/"
    web = WebURL(web_url)
    web.save_transcript_to_file()

if __name__ == "__main__":
    main()
