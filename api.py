from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from langgraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)


ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = ROOT / "frontend" / "dist"
RESUME_FILE = ROOT / "frontend" / "public" / "Vivek_AI_Enginner.pdf"
MAX_PDF_BYTES = 25 * 1024 * 1024

app = FastAPI(title="LangGraph RAG Chatbot", version="1.0.0")

if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    return str(content)


def _thread_state(thread_id: str) -> list[Any]:
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def _thread_summary(thread_id: str) -> dict[str, Any]:
    messages = _thread_state(thread_id)
    first_user_message = next(
        (
            _content_text(message.content)
            for message in messages
            if isinstance(message, HumanMessage)
        ),
        "New chat",
    )
    title = " ".join(first_user_message.split()) or "New chat"
    if len(title) > 48:
        title = f"{title[:47].rstrip()}…"

    return {
        "id": thread_id,
        "title": title,
        "document": thread_document_metadata(thread_id),
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/resume")
def download_resume() -> FileResponse:
    if not RESUME_FILE.is_file():
        raise HTTPException(status_code=404, detail="Resume file not found.")
    return FileResponse(
        RESUME_FILE,
        media_type="application/pdf",
        filename="Vivek_AI_Enginner.pdf",
    )


@app.get("/api/threads")
def list_threads() -> list[dict[str, Any]]:
    thread_ids = retrieve_all_threads()
    summaries = [_thread_summary(thread_id) for thread_id in thread_ids]
    return [
        summary
        for summary in sorted(summaries, key=lambda row: row["title"].lower())
        if summary["title"] != "New chat"
    ]


@app.post("/api/threads")
def create_thread() -> dict[str, str]:
    return {"id": str(uuid4()), "title": "New chat"}


@app.get("/api/threads/{thread_id}")
def get_thread(thread_id: str) -> dict[str, Any]:
    messages = []
    for message in _thread_state(thread_id):
        if isinstance(message, HumanMessage):
            messages.append({"role": "user", "content": _content_text(message.content)})
        elif isinstance(message, AIMessage) and message.content:
            messages.append({"role": "assistant", "content": _content_text(message.content)})

    return {
        "id": thread_id,
        "messages": messages,
        "document": thread_document_metadata(thread_id),
    }


@app.post("/api/threads/{thread_id}/pdf")
async def upload_pdf(thread_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please choose a PDF file.")

    contents = await file.read(MAX_PDF_BYTES + 1)
    if len(contents) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF must be 25 MB or smaller.")
    if not contents:
        raise HTTPException(status_code=400, detail="The selected PDF is empty.")

    try:
        result = await run_in_threadpool(
            ingest_pdf,
            contents,
            thread_id,
            file.filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not index PDF: {exc}") from exc

    return result


@app.post("/api/threads/{thread_id}/messages")
async def send_message(thread_id: str, request: ChatRequest) -> dict[str, str]:
    message_text = request.message.strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = await run_in_threadpool(
            chatbot.invoke,
            {"messages": [HumanMessage(content=message_text)]},
            {"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"The assistant could not respond: {exc}") from exc

    answer = next(
        (
            _content_text(message.content)
            for message in reversed(result.get("messages", []))
            if isinstance(message, AIMessage) and message.content
        ),
        "I couldn't generate a response. Please try again.",
    )
    return {"answer": answer}


@app.get("/")
def frontend() -> FileResponse:
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.is_file():
        raise HTTPException(
            status_code=503,
            detail="Build the React app with `npm run build` in the frontend folder.",
        )
    return FileResponse(index_file)
