from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
import json

class WebURL:
    def __init__(self, url: str):
        self.url = url
        self.metadata = self.get_webpage_metadata(url)
        
        if not self.metadata:
            self.title = None
            self.description = None
            self.publish_date = None
            self.view_count = 0
            self.like_count = 0
            self.dislike_count = 0
            self.comment_count = 0
        else:
            self.title = self.metadata.get("title")
            self.description = self.metadata.get("description")
            self.publish_date = self.metadata.get("publish_date", "")
            self.view_count = self.metadata.get("view_count", 0)
            self.like_count = self.metadata.get("like_count", 0)
            self.dislike_count = self.metadata.get("dislike_count", 0)
            self.comment_count = self.metadata.get("comment_count", 0)

    def get_webpage_metadata(self, url: str) -> dict:
        """
        Fetches metadata for the webpage.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('title').text if soup.find('title') else 'No title'
            description = soup.find('meta', attrs={'name': 'description'})
            description = description['content'] if description else 'No description'
            
            metadata = {
                "id": url,  # Ensure id and uri are the same
                "uri": url,
                "source_type": "web",
                "title": title,
                "description": description,
                "publish_date": "",  # Default to empty string
                "view_count": 0,     # Default to zero
                "like_count": 0,     # Default to zero
                "dislike_count": 0,  # Default to zero
                "comment_count": 0   # Default to zero
            }
            return metadata
        except Exception as e:
            print(f"An error occurred while fetching webpage metadata: {e}")
            return None

    def download_webpage_transcript(self) -> str:
        """
        Downloads the transcript for the webpage.
        """
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            transcript = soup.get_text()
            return transcript
        except Exception as e:
            print(f"An error occurred while downloading the webpage transcript: {e}")
            return ""

    def save_metadata_to_file(self, meta_file_path: str):
        """
        Saves the metadata to a file in JSON format.
        """
        with open(meta_file_path, "w") as meta_file:
            json.dump(self.metadata, meta_file, indent=4)
        print(f"Metadata saved to: {meta_file_path}")

    def save_transcript_to_file(self):
        """
        Saves the webpage transcript and metadata to files.
        """
        file_path = os.path.join("transcripts", f"{self.title}.txt")
        meta_file_path = os.path.join("transcripts", f"META_{self.title}.json")
        transcript = self.download_webpage_transcript()
        
        if transcript:
            with open(file_path, "w") as file:
                file.write(transcript)
            print(f"Transcript saved to: {file_path}")
            
            self.save_metadata_to_file(meta_file_path)
        else:
            print("No transcript to save.")

def main():
    load_dotenv()
    url = "https://finance.yahoo.com/news/live/stock-market-today-dow-pops-nasdaq-slips-as-focus-turns-to-cpi-inflation-report-210216764.html"  # Replace with the actual URL
    web_url = WebURL(url)
    
    print(f"Title: {web_url.title}")
    print(f"Description: {web_url.description}")
    
    web_url.save_transcript_to_file()

if __name__ == "__main__":
    main()