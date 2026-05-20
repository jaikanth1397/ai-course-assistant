from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    selected_video: str | None = None


@app.get("/")
def home():
    return {
        "message": "AI Course Assistant Backend"
    }


@app.post("/chat/{course_id}")
async def chat(
    course_id: str,
    req: ChatRequest
):

    response = ask_question(
        question=req.question,
        course_id=course_id,
        selected_video=req.selected_video
    )

    return response