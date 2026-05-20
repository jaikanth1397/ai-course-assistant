import os
import chromadb

from openai import OpenAI
from dotenv import load_dotenv
from moviepy import VideoFileClip

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

load_dotenv()

client = OpenAI()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="course_content"
)


def ingest_course(course_id):

    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    course_folder = os.path.join(
        BASE_DIR,
        "courses",
        course_id
    )

    if not os.path.exists(course_folder):

        raise Exception(
            f"No course folder found for {course_id}"
        )

    video_files = [
        file
        for file in os.listdir(course_folder)
        if file.endswith(".mp4")
    ]

    os.makedirs("audio", exist_ok=True)
    os.makedirs("transcripts", exist_ok=True)

    for video in video_files:

        print(f"Processing {video}")

        video_path = os.path.join(
            course_folder,
            video
        )

        video_name = os.path.splitext(
            video
        )[0]

        transcript_path = os.path.join(
            "transcripts",
            f"{course_id}_{video_name}.txt"
        )

        # -------------------------
        # USE EXISTING TRANSCRIPT
        # -------------------------

        if os.path.exists(transcript_path):

            print(
                f"Using existing transcript for {video}"
            )

            with open(
                transcript_path,
                "r"
            ) as f:

                transcript = f.read()

        else:

            audio_path = os.path.join(
                "audio",
                f"{video_name}.mp3"
            )

            # Extract audio
            clip = VideoFileClip(video_path)

            clip.audio.write_audiofile(
                audio_path
            )

            # Whisper transcription
            with open(audio_path, "rb") as audio_file:

                transcript_response = (
                    client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                )

            transcript = (
                transcript_response.text
            )

            with open(
                transcript_path,
                "w"
            ) as f:

                f.write(transcript)

        # -------------------------
        # CHUNKING
        # -------------------------

        chunks = splitter.split_text(
            transcript
        )

        # -------------------------
        # EMBEDDINGS
        # -------------------------

        for idx, chunk in enumerate(chunks):

            embedding_response = (
                client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
            )

            embedding = (
                embedding_response
                .data[0]
                .embedding
            )

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[
                    f"{course_id}_{video_name}_{idx}"
                ],
                metadatas=[{
                    "course_id": course_id,
                    "video_name": video_name,
                    "source": video
                }]
            )

        print(f"Finished {video}")

    print("Ingestion complete")