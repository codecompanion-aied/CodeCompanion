import os
import gc
import uuid
import subprocess
import streamlit as st
import time
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llms import LLMMetadata
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
import pandas as pd
import csv
import re
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

PATH_TO_COLLECTION = ""
PATH_TO_SAVE_EMBEDDINGS = ""
COLLECTION_NAME = "CodeCompanion"
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-l"
RERANK_MODEL = "BAAI/bge-reranker-large"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TOP_K = 3  # Number of most relevant k passages to retrieve
TOP_K_RERANK = 50 # Number of most relevant k passages to retrieve before reranking
COLLECTION_CREATED = True

client = chromadb.PersistentClient()

try:
    collection = client.get_collection(name=COLLECTION_NAME)
except ValueError:
    COLLECTION_CREATED = False
    print("Collection doesn't exist")

if not COLLECTION_CREATED:
    """
    Code here to implement indexing pipeline
    """
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(ids=[], embeddings=[], documents=[], metadatas=[])

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


with st.sidebar:
    start_button = st.button("Start CodeCompanion")
    if start_button:
        HF_TOKEN = st.text_input("HuggingFace Access Token")
        llm_container = st.empty()  # Placeholder for dynamic messages
        with st.spinner(f"Loading LLM: {LLM_MODEL}"):
            # Define HuggingFace tokens needed to access the model and the Inference API
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
            os.environ["HF_TOKEN"] = HF_TOKEN

            # Define the LLM model through the HuggingFace Inference API and set temperature to 0.01 to ensure near-deterministic behavior
            llm = HuggingFaceInferenceAPI(
                repo_id=LLM_MODEL, max_length=4096, temperature=0.01
            )
            llm_container.success(f"{llm_id} loaded successfully!")

        embed_container = st.empty()  # Placeholder for dynamic messages
        with st.spinner(f"Loading Embedding Model: {EMBED_MODEL}"):
            # Define embedding model
            embed_model = HuggingFaceInferenceAPIEmbedding(model_name=EMBED_MODEL)
            embed_container.success(f"{embed_id} loaded successfully!")

        rerank_container = st.empty()  # Placeholder for dynamic messages
        with st.spinner(f"Loading Reranker Model: {RERANK_MODEL}"):
            # Define embedding model
            rerank_model = HuggingFaceInferenceAPIEmbedding(model_name=RERANK_MODEL)
            rerank_container.success(f"{RERANK_MODEL} loaded successfully!")

        collection_container = st.empty()  # Placeholder for dynamic messages
        with st.spinner(f"Loading Collection: {COLLECTION_NAME}"):
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.node_postprocessor = rerank_model

            vector_store = ChromaVectorStore(chroma_collection=collection)
            index = VectorStoreIndex.from_vector_store(vector_store)

            collection_container.success(f"{COLLECTION_NAME} collection loaded successfully!")

        query_engine = index.as_query_engine(streaming=True, similarity_top_k=TOP_K)

        qa_prompt_tmpl_str = (
            """You are a helpful assistant whose role is to provide an answer to a user question based on the given context.\n"""
            """Answer the question exclusively based on the provided context.\n"""
            """Even if asked to, do not ever provide any python code or reference directly any python library, function or attribute.\n"""
            """Instead, reply with "I am sorry, but I cannot directly provide any code solution". Focus then on explaining the concepts and providing hints to help the user understand and implement the solution themselves.\n"""
            """Do not mention the context itself when replying.\n"""
            """User question: {question}\n"""
            """Context: {original_code}\n"""
            """Answer: """
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )

        st.session_state.query_engine = query_engine
        st.success("CodeCompanion is ready to Chat!")

col1, col2 = st.columns([6, 1])

with col1:
    st.header("CodeCompanion by Team CC #AI4Impact2024")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # context = st.session_state.context
        query_engine = st.session_state.query_engine

        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)

        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})