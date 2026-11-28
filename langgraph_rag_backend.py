from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()


# -------------------
# 1. LLM + embeddings
# -------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

@lru_cache(maxsize=1)
def _get_embeddings():
    """Load the local embedding model only when the first PDF is ingested."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -------------------
# 2. PDF retriever store (per thread)
# -------------------

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}
_ROOT = Path(__file__).resolve().parent
_RESUME_PATH = _ROOT / "frontend" / "public" / "Vivek_AI_Enginner.pdf"
_RESUME_TEXT: Optional[str] = None


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    return None


def _get_resume_text() -> Optional[str]:
    """Read and cache the short shared resume without needing embeddings."""
    global _RESUME_TEXT
    if _RESUME_TEXT is not None:
        return _RESUME_TEXT
    if not _RESUME_PATH.is_file():
        return None
    pages = PyPDFLoader(str(_RESUME_PATH)).load()
    _RESUME_TEXT = "\n\n".join(page.page_content.strip() for page in pages if page.page_content.strip())
    return _RESUME_TEXT or None


def is_roman_hinglish(text: str) -> bool:
    """Detect common Roman Hindi words mixed with English."""
    hindi_words = {
        "aap", "ap", "apka", "apki", "batao", "btao", "baare", "bare",
        "hai", "hain", "ho", "h", "kya", "ka", "ki", "ke", "mein",
        "me", "mera", "meri", "mere", "mujhe", "m", "nahi", "nahin",
        "aur", "se", "ko", "kaise", "kar", "karo", "kr", "sakte",
        "chahiye", "bta", "mujh", "kuch", "bhi", "kyu", "kyon",
    }
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    return len(words & hindi_words) >= 2


def _is_resume_query(text: str) -> bool:
    terms = (
        "resume", "cv", "education", "educational", "qualification", "degree",
        "college", "university", "school", "skills", "experience", "career",
        "work history", "profile", "certificate", "padhai", "shiksha", "kaam",
        "naukri", "रिज्यूमे", "शिक्षा", "पढ़ाई", "कौशल", "अनुभव", "डिग्री",
        "योग्यता", "नौकरी", "बायोडाटा",
    )
    lower_text = text.lower()
    return any(term in lower_text for term in terms)


def rewrite_answer_in_hinglish(question: str, answer: str) -> str:
    """Rewrite a Devanagari answer in the user's Roman Hinglish style."""
    response = llm.invoke([
        SystemMessage(content=(
            "Rewrite the draft answer in natural Roman Hinglish, matching the user's "
            "way of writing Hindi with Latin letters and English words. Use Latin "
            "letters only; do not output any Devanagari characters. Preserve all "
            "facts and useful Markdown structure. Do not add or remove facts. Never "
            "label the answer as Hinglish or put '(Hinglish)' in a heading."
        )),
        HumanMessage(content=f"User's question:\n{question}\n\nDraft answer:\n{answer}"),
    ])
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        ).strip()
    return str(content).strip()


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF
    and store it for the thread.
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(
            chunks,
            _get_embeddings()
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        thread_id = str(thread_id)

        _THREAD_RETRIEVERS[thread_id] = retriever

        _THREAD_METADATA[thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:

        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------

search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.

    Supported operations:
    add, sub, mul, div
    """

    try:

        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":

            if second_num == 0:
                return {
                    "error": "Division by zero is not allowed"
                }

            result = first_num / second_num

        else:
            return {
                "error": f"Unsupported operation '{operation}'"
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:

        return {
            "error": str(e)
        }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol
    using Alpha Vantage.
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {
            "error": "ALPHA_VANTAGE_API_KEY is not configured."
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
    )

    try:

        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:

        return {
            "error": str(e)
        }


@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None
) -> dict:
    """
    Retrieve relevant information from this chat's uploaded PDF and the owner's
    resume, which is available in every chat.

    Always include the thread_id when calling this tool.
    """

    context = []
    metadata = []
    source_files = []
    thread_retriever = _get_retriever(thread_id)
    if thread_retriever is not None:
        source = _THREAD_METADATA.get(str(thread_id), {}).get("filename", "Uploaded PDF")
        source_files.append(source)
        for doc in thread_retriever.invoke(query):
            context.append(f"Source: {source}\n{doc.page_content}")
            metadata.append({**doc.metadata, "source_file": source})

    if _is_resume_query(query):
        resume_text = _get_resume_text()
        if resume_text:
            source_files.append(_RESUME_PATH.name)
            context.append(f"Source: {_RESUME_PATH.name}\n{resume_text}")
            metadata.append({"source_file": _RESUME_PATH.name})

    if not context:
        return {
            "error": (
                "No resume or document is available. "
                "Upload a PDF first."
            ),
            "query": query,
        }

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_files": source_files,
    }


tools = [
    search_tool,
    get_stock_price,
    calculator,
    rag_tool,
]


llm_with_tools = llm.bind_tools(tools)


# -------------------
# 4. State
# -------------------

class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# -------------------
# 5. Nodes
# -------------------

def chat_node(
    state: ChatState,
    config=None
):
    """
    LLM node that may answer
    or request a tool call.
    """

    thread_id = None

    if config and isinstance(config, dict):

        thread_id = (
            config
            .get("configurable", {})
            .get("thread_id")
        )

    latest_user_message = next(
        (message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
        None,
    )
    latest_user_text = (
        latest_user_message.content
        if latest_user_message and isinstance(latest_user_message.content, str)
        else ""
    )

    system_content = (
            "You are a helpful AI assistant.\n\n"

            "Format responses in clean GitHub-flavored Markdown. Start with a direct "
            "answer or a brief summary, then organize details under clear headings. "
            "Use short paragraphs and concise bullet points, with one idea per bullet. "
            "For news and other grouped facts, use topic headings and bullets instead "
            "of a table. Do not use tables unless the user asks for one or a compact "
            "comparison truly needs it. Match the user's language and writing script. "
            "If the user writes Hindi in Latin letters (Hinglish), reply in natural "
            "Hinglish using Latin letters; do not switch to Devanagari. If the user "
            "writes Hindi in Devanagari, reply in Devanagari. If the user writes in "
            "English, reply in English. For mixed-language messages, follow the "
            "dominant language and script. Never include raw "
            "language labels such as '(Hinglish)' in headings or answers. "
            "HTML tags such as `<br>`.\n\n"

            "For questions about the uploaded PDF or the owner's resume, "
            "use the `rag_tool` and include the thread_id "
            f"`{thread_id}`. The owner's resume is available in every chat.\n\n"

            "You can use web search for current information, "
            "the stock price tool for stock information, "
            "and the calculator for arithmetic.\n\n"

            "If the user asks about a document, use the rag_tool before answering. "
            "The owner's resume is searchable even when no user PDF has been uploaded."
    )

    if is_roman_hinglish(latest_user_text):
        system_content += (
            "\n\nSTRICT SCRIPT RULE FOR THIS MESSAGE: The user wrote in Roman Hinglish. "
            "Write the entire answer in Roman Hinglish using Latin letters only. "
            "Do not use Devanagari anywhere, including headings or bullets."
        )

    if _is_resume_query(latest_user_text):
        resume_text = _get_resume_text()
        if resume_text:
            system_content += (
                "\n\nThe user is asking about the owner's resume. Use this extracted "
                "resume text as the source of truth and answer from it. Do not say "
                "the resume needs to be uploaded; it is already provided. If a detail "
                "is absent from this text, say that the resume does not specify it.\n"
                f"RESUME TEXT:\n{resume_text}"
            )

    system_message = SystemMessage(content=system_content)

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config
    )

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)


# -------------------
# 6. Checkpointer
# -------------------

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(
    conn=conn
)


# -------------------
# 7. Graph
# -------------------

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge(
    "tools",
    "chat_node"
)

chatbot = graph.compile(
    checkpointer=checkpointer
)


# -------------------
# 8. Helpers
# -------------------

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        all_threads.add(
            checkpoint.config[
                "configurable"
            ]["thread_id"]
        )

    return list(all_threads)


def thread_has_document(
    thread_id: str
) -> bool:

    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(
    thread_id: str
) -> dict:

    return _THREAD_METADATA.get(
        str(thread_id),
        {}
    )


def delete_thread(thread_id: str) -> None:
    """Delete a chat's saved checkpoints and in-memory PDF index."""
    thread_id = str(thread_id)
    checkpointer.delete_thread(thread_id)
    _THREAD_RETRIEVERS.pop(thread_id, None)
    _THREAD_METADATA.pop(thread_id, None)
