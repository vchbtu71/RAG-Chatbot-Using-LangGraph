# LangGraph RAG Chatbot

A document question-answering chatbot with a React interface, a FastAPI service, LangGraph tools, FAISS retrieval, Hugging Face embeddings, and Groq chat completion.

## Project layout

```text
chatbot_by_rag/
├── api.py                       # FastAPI endpoints and React static hosting
├── langgraph_rag_backend.py     # LangGraph, tools, PDF ingestion, SQLite history
├── frontend/                    # React + Vite user interface
├── chatbot.db                   # Local conversation checkpoints (created at runtime)
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python 3.12+
- Node.js 20.19+ and npm
- A Groq API key
- Internet access for Groq requests and the first download of the embedding model

Create a `.env` file in this directory:

```env
GROQ_API_KEY=your_groq_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
```

The Alpha Vantage key is optional unless you want to use stock lookup.

## Install

From the project folder, install the Python dependencies:

```powershell
uv sync
```

Install the frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## Run for development

Open two terminals in the project folder.

Terminal 1, start the API:

```powershell
uv run uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2, start the React development server:

```powershell
cd frontend
npm run dev
```

Open the frontend URL printed by Vite, usually `http://localhost:5173`.

## Run as one local app

Build the React interface:

```powershell
cd frontend
npm run build
cd ..
```

Then start FastAPI:

```powershell
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`.

## Workflow

1. The React app sends chat, PDF upload, and conversation-history requests to FastAPI.
2. FastAPI passes chat requests to the LangGraph backend and sends PDF bytes to `ingest_pdf`.
3. PDF ingestion extracts text, splits it into chunks, creates embeddings, and builds a FAISS retriever for that chat.
4. LangGraph selects PDF retrieval, web search, calculator, or stock lookup as needed.
5. SQLite stores conversation checkpoints; FastAPI returns messages to React for display.

The embedding model loads on the first PDF ingestion and stays cached for the life of the API process. FAISS document indexes currently stay in memory, so re-upload documents after restarting the API.

## API

- `GET /api/health`
- `GET /api/threads`
- `POST /api/threads`
- `GET /api/threads/{thread_id}`
- `POST /api/threads/{thread_id}/pdf`
- `POST /api/threads/{thread_id}/messages`
