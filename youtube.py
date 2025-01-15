import os
import json
from typing import List, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

load_dotenv()

class Youtube:

    def __init__(self, api_key: str, youtube_url: str):
        """
        Initializes the Youtube class with API key and YouTube URL.
        """
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.youtube_url = youtube_url
        self.youtube_id = self.extract_video_id(youtube_url)
        self.metadata = self.get_video_metadata()
        
        if not self.metadata:
            self.title = None
            self.description = None
            self.publish_date = None
            self.view_count = None
            self.like_count = None
            self.dislike_count = None
            self.comment_count = None
        else:
            self.title = self.metadata.get("title")
            self.description = self.metadata.get("description")
            self.publish_date = self.metadata.get("publish_date")
            self.view_count = self.metadata.get("view_count")
            self.like_count = self.metadata.get("like_count")
            self.dislike_count = self.metadata.get("dislike_count")
            self.comment_count = self.metadata.get("comment_count")

    def get_video_metadata(self) -> dict:
        """
        Fetches metadata for the YouTube video.
        """
        try:
            video_id = self.youtube_id
            request = self.youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            )
            response = request.execute()
            if not response["items"]:
                return None

            video_info = response["items"][0]
            snippet = video_info["snippet"]
            statistics = video_info["statistics"]

            metadata = {
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "publish_date": snippet.get("publishedAt"),
                "view_count": int(statistics.get("viewCount", 0)),
                "like_count": int(statistics.get("likeCount", 0)),
                "dislike_count": int(statistics.get("dislikeCount", 0)),
                "comment_count": int(statistics.get("commentCount", 0))
            }
            return metadata
        except Exception as e:
            print(f"An error occurred while fetching video metadata: {e}")
            return None

    def download_youtube_transcript(self) -> str:
        """
        Downloads the transcript for the YouTube video using youtube_transcript_api.
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.youtube_id)
            transcript = transcript_list.find_transcript(['en'])
            transcript_pieces = transcript.fetch()
            transcript_text = ' '.join([piece['text'] for piece in transcript_pieces])
            return transcript_text
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            print(f"An error occurred while fetching the YouTube transcript: {e}")
            return ""
        except Exception as e:
            print(f"An error occurred: {e}")
            return ""

    def save_transcript_to_file(self):
        """
        Saves the downloaded transcript and metadata to files.
        """
        file_path = os.path.join("transcripts", f"{self.title}.txt")
        meta_file_path = os.path.join("transcripts", f"META_{self.title}.json")
        
        if os.path.exists(file_path):
            print(f"Transcript already exists: {file_path}")
            return
        else:
            print("Downloading the Transcript!")
        
        transcript = self.download_youtube_transcript()
        
        if transcript:
            with open(file_path, "w") as file:
                file.write(transcript)
            print(f"Transcript saved to: {file_path}")
            
            self.save_metadata_to_file(meta_file_path)
        else:
            print("No transcript to save.")

    def save_metadata_to_file(self, meta_file_path: str):
        """
        Saves the metadata to a file in JSON format.
        """
        with open(meta_file_path, "w") as meta_file:
            json.dump(self.metadata, meta_file, indent=4)
        print(f"Metadata saved to: {meta_file_path}")

    def extract_video_id(self, youtube_url: str) -> str:
        """
        Extracts the video ID from the YouTube URL.
        """
        video_id = youtube_url.split("v=")[1]
        return video_id

def main():
    api_key = os.getenv("YOUTUBE_API_KEY")
    youtube_url = "https://www.youtube.com/watch?v=zm0QVutAkYg"
    
    yt = Youtube(api_key, youtube_url)
    
    # Download and save transcript
    yt.save_transcript_to_file()

if __name__ == "__main__":
    main()