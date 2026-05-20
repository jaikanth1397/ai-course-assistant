# AI Course Assistant using RAG

An AI-powered learning assistant that allows students to ask questions directly from course videos using Retrieval-Augmented Generation (RAG).

---

# 🚀 Features

- AI-powered Q&A assistant
- Semantic search using embeddings
- Video transcription using Whisper
- ChromaDB vector storage
- Next.js frontend + FastAPI backend

---

# 🛠️ Tech Stack

## Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS

## Backend
- FastAPI
- Python
- OpenAI API
- ChromaDB
- Whisper API

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-course-assistant.git
cd ai-course-assistant
```

## Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# 📹 Adding Videos

Place videos inside:

```text
backend/courses/AI-Course/videos/
```

---

# 🤖 How RAG Works

1. Extract audio from videos
2. Generate transcript using Whisper
3. Split transcript into chunks
4. Create embeddings
5. Store vectors in ChromaDB
6. Retrieve relevant chunks
7. Generate AI answer using GPT

---

# 👨‍💻 Author

AI Engineering Capstone Project

---

# 📜 License

MIT License
