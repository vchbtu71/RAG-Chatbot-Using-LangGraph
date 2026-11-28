from __future__ import annotations

import os
import hashlib
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

from langchain_core.messages import BaseMessage, SystemMessage
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
_RESUME_INDEX_PATH = _ROOT / ".data" / "resume_faiss"
_RESUME_HASH_PATH = _ROOT / ".data" / "resume.sha256"
_RESUME_RETRIEVER: Any = None


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    return None


def _get_resume_retriever():
    """Lazily load or build the shared resume index and cache it between runs."""
    global _RESUME_RETRIEVER
    if _RESUME_RETRIEVER is not None:
        return _RESUME_RETRIEVER
    if not _RESUME_PATH.is_file():
        return None

    resume_hash = hashlib.sha256(_RESUME_PATH.read_bytes()).hexdigest()
    index_exists = (
        (_RESUME_INDEX_PATH / "index.faiss").is_file()
        and (_RESUME_INDEX_PATH / "index.pkl").is_file()
        and _RESUME_HASH_PATH.is_file()
        and _RESUME_HASH_PATH.read_text(encoding="utf-8").strip() == resume_hash
    )

    if index_exists:
        vector_store = FAISS.load_local(
            str(_RESUME_INDEX_PATH),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    else:
        docs = PyPDFLoader(str(_RESUME_PATH)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source"] = _RESUME_PATH.name
        vector_store = FAISS.from_documents(chunks, _get_embeddings())
        _RESUME_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(_RESUME_INDEX_PATH))
        _RESUME_HASH_PATH.write_text(resume_hash, encoding="utf-8")

    _RESUME_RETRIEVER = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    return _RESUME_RETRIEVER


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

    retrievers = []
    thread_retriever = _get_retriever(thread_id)
    resume_retriever = _get_resume_retriever()
    if thread_retriever is not None:
        source = _THREAD_METADATA.get(str(thread_id), {}).get("filename", "Uploaded PDF")
        retrievers.append((thread_retriever, source))
    if resume_retriever is not None:
        retrievers.append((resume_retriever, _RESUME_PATH.name))

    if not retrievers:
        return {
            "error": (
                "No resume or document is available. "
                "Upload a PDF first."
            ),
            "query": query,
        }

    context = []
    metadata = []
    for retriever, source_file in retrievers:
        for doc in retriever.invoke(query):
            context.append(f"Source: {source_file}\n{doc.page_content}")
            metadata.append({**doc.metadata, "source_file": source_file})

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_files": [source for _, source in retrievers],
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

    system_message = SystemMessage(
        content=(
            "You are a helpful AI assistant.\n\n"

            "Format responses in clean GitHub-flavored Markdown. Start with a direct "
            "answer or a brief summary, then organize details under clear headings. "
            "Use short paragraphs and concise bullet points, with one idea per bullet. "
            "For news and other grouped facts, use topic headings and bullets instead "
            "of a table. Do not use tables unless the user asks for one or a compact "
            "comparison truly needs it. Match the user's language. Never include raw "
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
    )

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
