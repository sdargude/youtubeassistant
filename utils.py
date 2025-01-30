import json

def read_transcript(transcript_path: str):
    with open(transcript_path, "r") as file:
        transcript = file.read()
    return transcript

def read_metadata(metadata_path: str):
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    return metadata
