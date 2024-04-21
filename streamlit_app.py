import os
import getpass
import gc
import re
import uuid
import textwrap
import subprocess
import nest_asyncio
from dotenv import load_dotenv

import streamlit as st

from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.readers.github import GithubRepositoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext

from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)


# Input for Jina API key
text_input_container = st.empty()
text_input_container.text_input("Jina API key", key="api_key", type="password")

if st.session_state.api_key != "":
    text_input_container.empty()
    jinaai_api_key = st.session_state.api_key
    # setting up the embedding model
    embed_model = JinaEmbedding(
        api_key=jinaai_api_key,
        model="jina-embeddings-v2-base-code",
    )

# setting up the llm
llm=HuggingFaceInferenceAPI(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
)


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


owner = "larymak"
repo = "Python-project-Scripts"
branch = "main"

message_container = st.empty()  # Placeholder for dynamic messages

with st.spinner(f"Loading {repo} repository by {owner}..."):
    documents = GithubRepositoryReader(
        owner=owner,
        repo=repo,
        use_parser=False,
        verbose=False,
        ignore_directories=["examples"],
    ).load_data(branch=branch)

    docs = loader.load_data()

    # ====== Create vector store and upload data ======
    Settings.embed_model = embed_model
    index = VectorStoreIndex.from_documents(docs)

    # ====== Setup a query engine ======
    Settings.llm = llm
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
    
    # ====== Customise prompt template ======
    qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
    "Query: {query_str}\n"
    "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )

    if docs:
        message_container.success("Data loaded successfully!!")
    else:
        message_container.write(
            "No data found, check if the repository is not empty!"
        )
    st.session_state.query_engine = query_engine

    st.success("Ready to Chat!")

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with your code! </>")

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
if prompt := st.chat_input("What's up?"):
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
