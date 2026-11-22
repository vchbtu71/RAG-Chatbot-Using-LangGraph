## 📌 Project Overview

A **LangGraph-powered RAG chatbot** that lets users upload PDF documents and interact with them through natural-language queries. The system combines **FAISS vector search and Hugging Face embeddings** for document retrieval with **Groq LLMs** for response generation, while LangGraph orchestrates tool calling and conversation flow. The Streamlit interface provides an interactive chat experience with support for PDF-based Q&A, web search, calculations, and stock-price queries.


# 🤖 LangGraph RAG Chatbot

A document-based **RAG (Retrieval-Augmented Generation) chatbot** built using **LangGraph, LangChain, Groq, FAISS, Hugging Face embeddings, and Streamlit**.

The application allows users to upload a PDF and ask questions about its content. The system retrieves relevant document chunks and uses an LLM to generate context-aware answers.

## 🚀 Features

* 📄 Upload and process PDF documents
* 🔍 Semantic document retrieval using FAISS
* 🧠 Hugging Face sentence-transformer embeddings
* 🤖 LLM-powered responses using Groq
* 🔗 LangGraph-based agent workflow
* 🛠️ Tool calling support
* 🌐 Web search using DuckDuckGo
* 🧮 Calculator tool
* 📈 Stock price lookup using Alpha Vantage
* 💬 Persistent conversation history using SQLite
* 🖥️ Interactive Streamlit frontend
* 🧵 Separate conversation threads for different chats

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     LangGraph       │
                    │   Agent Workflow    │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
            PDF RAG       Web Search     Calculator
                 │
                 ▼
        ┌───────────────────┐
        │ Hugging Face      │
        │ Embeddings        │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │       FAISS       │
        │   Vector Store    │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │      Groq LLM     │
        └─────────┬─────────┘
                  │
                  ▼
              Response
```

## 📂 Project Structure

```text
chatbot_by_rag/
│
├── langgraph_rag_backend.py
├── streamlit_rag_frontend.py
├── rag_by_langgraph.py
│
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

### Backend

`langgraph_rag_backend.py`

Contains:

* PDF ingestion
* Text splitting
* Embedding generation
* FAISS vector store
* RAG retrieval
* LangGraph workflow
* Tool definitions
* Groq LLM configuration
* SQLite conversation checkpointing

### Frontend

`streamlit_rag_frontend.py`

Provides the Streamlit interface for:

* Creating conversations
* Uploading PDFs
* Asking questions
* Viewing previous conversations
* Streaming model responses

## 🛠️ Tech Stack

| Technology    | Purpose                              |
| ------------- | ------------------------------------ |
| Python        | Core programming language            |
| LangGraph     | Agent/workflow orchestration         |
| LangChain     | RAG and LLM integration              |
| Groq          | LLM inference                        |
| Hugging Face  | Text embeddings                      |
| FAISS         | Vector similarity search             |
| Streamlit     | Frontend                             |
| SQLite        | Conversation persistence             |
| PyPDF         | PDF processing                       |
| DuckDuckGo    | Web search                           |
| Alpha Vantage | Stock market data                    |
| uv            | Python package/dependency management |

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ianmol94/chatbot_by_rag.git
cd chatbot_by_rag
```

### 2. Install dependencies

This project uses `uv`.

```bash
uv sync
```

Or install the required packages manually:

```bash
uv add langchain-groq \
langchain-huggingface \
langchain-community \
langchain-text-splitters \
langgraph \
langgraph-checkpoint-sqlite \
faiss-cpu \
sentence-transformers \
duckduckgo-search \
ddgs \
pypdf \
python-dotenv \
requests \
streamlit
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
```

Never commit your `.env` file.

Add it to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
chatbot.db
```

## ▶️ Run the Application

Start the Streamlit application with:

```bash
uv run streamlit run streamlit_rag_frontend.py
```

The application will be available at:

```text
http://localhost:8501
```

## 📄 How RAG Works

When a PDF is uploaded:

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Extraction
 ↓
Recursive Character Text Splitting
 ↓
Hugging Face Embeddings
 ↓
FAISS Vector Store
 ↓
Retriever
```

When the user asks a question:

```text
User Question
      ↓
LangGraph Agent
      ↓
RAG Tool
      ↓
FAISS Retriever
      ↓
Relevant Document Chunks
      ↓
Groq LLM
      ↓
Generated Answer
```

The model therefore answers PDF-related questions using retrieved document context instead of relying only on its pretrained knowledge.

## 🧠 LangGraph Workflow

The chatbot uses a graph-based workflow consisting of:

```text
START
  ↓
Chat Node
  ↓
Tool Decision
  ↓
Tool Node
  ↓
Chat Node
  ↓
END
```

The agent can decide when to use available tools such as:

* PDF retrieval
* Web search
* Calculator
* Stock price lookup

## 💬 Example Queries

After uploading a PDF, you can ask:

```text
What is the main topic of this document?

Summarize chapter 2.

What methodology was used in the research?

What are the key conclusions?

Compare the approaches discussed in the document.
```

You can also use general-purpose tools for queries such as:

```text
What is 125 * 48?

Search the web for the latest information about ...

What is the current stock price of ...
```

## 🔐 Security

API keys are loaded from environment variables:

```python
os.getenv("GROQ_API_KEY")
```

Do not hard-code API keys inside Python files or commit them to GitHub.

## ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud**.

Set the following secrets in the deployment settings:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Set the Streamlit entry point to:

```text
streamlit_rag_frontend.py
```

## ⚠️ Current Limitations

* FAISS indexes are stored in memory for the active application process.
* Uploaded documents are associated with individual conversation threads.
* SQLite storage is local to the deployment environment.
* Documents may need to be uploaded again after application restarts or redeployments.
* This architecture is primarily suited to a portfolio/demo application rather than a large-scale production system.

## 🔮 Future Improvements

* [ ] Replace in-memory FAISS with Qdrant
* [ ] Add persistent document storage
* [ ] Add user authentication
* [ ] Support multiple document uploads
* [ ] Add document management
* [ ] Improve citation/source display
* [ ] Add streaming token-level responses
* [ ] Add conversation deletion
* [ ] Add production-grade persistent database
* [ ] Improve retrieval using hybrid search
* [ ] Add reranking for retrieved documents

## 👨‍💻 Author

Off Campus Code : BNWR4Q

**Anmol Singh**

B.Tech Electronics Engineering
Harcourt Butler Technical University, Kanpur

GitHub: `https://github.com/ianmol94`

---

⭐ If you found this project useful, consider giving it a star.
