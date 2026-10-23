import os
from dotenv import load_dotenv

load_dotenv()

# Import libraries and packages
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.tools import tool
from langchain.agents import create_agent

import streamlit as st

# Streamlit Session State Initialization
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def document_process(path):
    # Load documents from directory
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = splitter.split_documents(documents=docs)

    # Create local embeddings and vector store (Fixed argument to 'embedding')
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = InMemoryVectorStore.from_documents(
        documents=splitted_docs, embedding=embeddings
    )

    # Initialize LLM with a valid Groq model
    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve documents relevant to a query from the knowledge base."""
        context = ""
        matched_docs = vector_db.similarity_search(query=query, k=3)
        for splitted_doc in matched_docs:
       
            context += splitted_doc.page_content + "\n\n"
        return context

    system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
    My knowledge base consists of the details from the uploaded document. 
    ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory,
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# Streamlit UI Flow
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(
        label="Select PDF Files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded:
        with st.spinner("Processing documents..."):
            path = "./doc_files/"
            os.makedirs(
                path, exist_ok=True
            ) 
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            document_process(path)
            st.rerun()

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask Anything related to uploaded documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": 1}},
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role": "ai", "content": answer})