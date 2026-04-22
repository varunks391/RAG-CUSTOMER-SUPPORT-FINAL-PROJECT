import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# ---------------- UI ----------------
st.set_page_config(page_title="Free RAG Bot", layout="centered")
st.title("💬 Free RAG Customer Support Assistant")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# ---------------- Main ----------------
if uploaded_file:

    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # FREE Embeddings
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(chunks, embedding)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # FREE LLM (small model)
    llm = pipeline("text-generation", model="distilgpt2")

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer using context.

    Context:
    {context}

    Question:
    {question}
    """)

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
        escalate: bool

    def retrieve(state):
        docs = retriever.invoke(state["question"])
        return {"context": docs}

    def process(state):
        context = "\n\n".join([d.page_content for d in state["context"]])

        input_text = f"Context: {context}\nQuestion: {state['question']}\nAnswer:"

        response = llm(input_text, max_length=200)[0]["generated_text"]

        return {"answer": response, "escalate": False}

    def output(state):
        return state

    # Graph
    graph = StateGraph(State)

    graph.add_node("retrieve", retrieve)
    graph.add_node("process", process)
    graph.add_node("output", output)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "process")
    graph.add_edge("process", "output")
    graph.add_edge("output", END)

    app = graph.compile()

    st.success("✅ FREE System Ready")

    query = st.text_input("Ask your question:")

    if query:
        result = app.invoke({"question": query})
        st.write("### Answer:")
        st.write(result["answer"])

else:
    st.info("Upload a PDF to start")