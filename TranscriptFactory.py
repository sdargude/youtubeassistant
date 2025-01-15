from abc import ABC, abstractmethod
from youtube import Youtube
from weburl import WebURL
from dotenv import load_dotenv
import os

load_dotenv()

class Transcript(ABC):
    @abstractmethod
    def save_transcript_to_file(self):
        pass

    @abstractmethod
    def save_metadata_to_file(self, meta_file_path: str):
        pass

class TranscriptFactory:
    @staticmethod
    def create_transcript(url: str) -> Transcript:
        if "youtube.com" in url or "youtu.be" in url:
            api_key = os.getenv("YOUTUBE_API_KEY")
            return Youtube(api_key, url)
        else:
            return WebURL(url)

def main():
    url = "https://www.youtube.com/watch?v=lh5Wj6QhbbU"
    transcript = TranscriptFactory.create_transcript(url)
    transcript.save_transcript_to_file()

if __name__ == "__main__":
    main()
