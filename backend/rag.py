import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from ingest import ingest_course

load_dotenv()

client = OpenAI()

# Chroma client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Collection
collection = chroma_client.get_or_create_collection(
    name="course_content"
)


def course_exists(course_id):

    results = collection.get(
        where={
            "course_id": course_id
        }
    )

    return len(results["ids"]) > 0


def ask_question(
    question,
    course_id,
    selected_video=None
):

    # Auto ingest if course not found
    if not course_exists(course_id):

        print(
            f"{course_id} not found. Starting ingestion..."
        )

        ingest_course(course_id)

    # Create embedding for question
    embedding_response = (
        client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        )
    )

    query_embedding = (
        embedding_response
        .data[0]
        .embedding
    )

    # ChromaDB metadata filtering
    if (
        selected_video
        and selected_video != "all"
    ):

        where_clause = {
            "$and": [
                {
                    "course_id": course_id
                },
                {
                    "video_name": selected_video
                }
            ]
        }

    else:

        where_clause = {
            "course_id": course_id
        }

    # Vector search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where=where_clause
    )

    retrieved_chunks = (
        results["documents"][0]
    )

    metadatas = (
        results["metadatas"][0]
    )

    # Build context
    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are a helpful AI course assistant.

Answer ONLY from the provided context.

If answer is not found in context,
say:
"I could not find that in the course."

Context:
{context}

Question:
{question}
"""

    # LLM response
    response = (
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    # Sources
    sources = []

    for meta in metadatas:

        sources.append({
            "video_name": meta["video_name"],
            "source": meta["source"]
        })

    return {
        "answer": answer
    }