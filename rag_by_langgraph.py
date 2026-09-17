import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from typing import Annotated, TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.tools import tool

from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0
)


## loading the doc
loader = PyPDFLoader("intro-to-ml.pdf")
docs = loader.load()

len(docs)

## splitter 
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
chunks = splitter.split_documents(docs)

len(chunks)


## embedding generation
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


## vector format
vector_store = FAISS.from_documents(chunks, embeddings)

# retriever 
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {'k':4})


## using LANGGRAPH now 
@tool 
def rag_tool(query):
    """
    Retrieve relevant information from the pdf document.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored documents.
    """
    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        'query': query,
        'context': context,
        'metadata': metadata
    }
    
# we are binding that LLM has a tool 
tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)

## LANGGRAPH starts now 
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


## state defintion in Langgraph
def chat_node(state:ChatState):

    messages = state['messages']

    response = llm_with_tools.invoke(messages)

    return {'messages': [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)



tool_node = ToolNode(tools)

graph = StateGraph(ChatState)


## node definition in LANGGRAPH
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

## edge definition in LANGGRAPH 
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile()


result = chatbot.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Using the pdf notes, explain how to find the ideal value of K in KNN"
                )
            )
        ]
    }
)

print(result['messages'][-1].content)