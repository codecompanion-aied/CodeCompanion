import os
import uuid
import csv
import streamlit as st
from streamlit_ace import st_ace
import chromadb
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from FlagEmbedding import FlagReranker
from langchain_community.llms import HuggingFaceEndpoint

COLLECTION_NAME = "CodeCompanion"
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-l"
RERANK_MODEL = "BAAI/bge-reranker-large"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TOP_K = 3  # Number of most relevant k passages to retrieve
TOP_K_RERANK = 50 # Number of most relevant k passages to retrieve before reranking


def reset_chat():
    st.session_state.messages = []


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
session_id = st.session_state.id
client = chromadb.PersistentClient()

if "messages" not in st.session_state:
    reset_chat()

if "token" not in st.session_state:
    st.session_state.token = False

if "choice" not in st.session_state:
    st.session_state.choice = False

if "project" not in st.session_state:
    st.session_state.project = None

if "task" not in st.session_state:
    st.session_state.task = 0

if "feedback" not in st.session_state:
    st.session_state.feedback = True

if "submission" not in st.session_state:
    st.session_state.submission = None

st.title("CodeCompanion by Team CC #AI4Impact2024")


@st.cache_resource(show_spinner=False)
def load_llm_model():
    llm = HuggingFaceEndpoint(repo_id=LLM_MODEL, max_length=4096, temperature=0.01)

    return llm


@st.cache_resource(show_spinner=False)
def load_embed_model(token):
    embed_model = HuggingFaceInferenceAPIEmbeddings(api_key=token, model_name=EMBED_MODEL)

    return embed_model


@st.cache_resource(show_spinner=False)
def load_rerank_model():
    rerank_model = FlagReranker(RERANK_MODEL, use_fp16=True)

    return rerank_model


def parse_corpus():
    '''
    Takes as input a path to folder containing a csv of Science lessons
    Returns a list of code and passages

     Args:
        path_to_files (str): path to folder containg files

    Returns
        project_descriptions (list): list of project descriptions
        project_meta (list): list of metadata
    '''

    project_descriptions = []
    project_meta = []
    with open("CodeCompanion - Test.csv") as f:
        data = csv.reader(f, delimiter=",")
        idx = 0
        for row in data:
            if idx == 0:
                idx = 1
                continue
            project_description = ' '.join(row[0:4])
            project_descriptions.append(project_description)
            meta = {"topic": row[1], "description": row[2], "structure": row[3], "steps": row[4]}
            for i in range(int(row[4])):
                meta[f"code_{i}"] = row[5 + 2 * i]
            for i in range(int(row[4])):
                meta[f"task_{i}"] = row[6 + 2 * i]
            project_meta.append(meta)

    print("# descr: ", len(project_descriptions))
    print("# meta: ", len(project_meta))

    return project_descriptions, project_meta


def generate_embeddings(passages):
    '''
    Takes a list of passages and generates embeddings

    Args:
      passages (list): list of passages

    Returns:
      passage_embeddings (list): list of embeddings
    '''

    passages_embeddings = embed_model.embed_documents(passages)

    print("# embed: ", len(passages_embeddings))
    print("Dim embed 1: ", len(passages_embeddings[0]))
    print("Dim embed 2: ", len(passages_embeddings[1]))
    print("Dim embed 3: ", len(passages_embeddings[2]))

    return passages_embeddings


def index_passages(descriptions, embeddings, metadatas):
    '''
    Takes as input meta and embeddings of the passages and puts in vector store
    Saves to disk

    Args:
      embeddings (list): list of embeddings
      meta (list): list of meta
    '''

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[str(i) for i in list(range(len(embeddings)))],
        documents=descriptions,
        embeddings=embeddings,
        metadatas=metadatas
    )


def retrieve_passages(query, top_k):
  '''
  Takes as input a query and finds the top k passages and returns them

  Args:
    query (str): query
    top_k (int): number of most relevant passages to be returned

  Returns
    metadatas (list): metadata corresponding to k most relevant passages
  '''

  collection = client.get_collection(name=COLLECTION_NAME)

  query_embed = embed_model.embed_query(query)
  print("Dim query embed: ", len(query_embed))

  results = collection.query(
    query_embeddings=query_embed,
    n_results=top_k,
    include = ["documents", "metadatas"]
  )

  descriptions = results['documents'][0]
  metadatas = results['metadatas'][0]

  return descriptions, metadatas


def rerank_passages(query):
  '''
  Takes as input a query, finds the top n passages, reranks them, and returns the top k

  Args:
    query (str): query

  Returns
    reranked_passages (list): k most relevant, reranked passages
    reranked_metadata (list): metadata corresponding to k most relevant, reranked passages
    reranked_scores (list): list of corresponding normalized inner product scores
  '''

  descriptions, metadatas = retrieve_passages(query, TOP_K_RERANK)

  scores = rerank_model.compute_score([[query, description] for description in descriptions])
  scores = [round(100 * score, 1) for score in scores]
  print("Len scores: ", len(scores))

  reranked_metadatas = []
  for metadata, score in zip(metadatas, scores):
    reranked_metadatas.append({"metadata": metadata, "score": score})

  reranked_metadatas = sorted(reranked_metadatas, key=lambda x: x["score"], reverse=True)
  reranked_metadatas = reranked_metadatas[0:TOP_K]

  reranked_metadata = []
  for meta in reranked_metadatas:
    reranked_metadata.append(meta["metadata"])

  return reranked_metadata


def generate_chat_answer(question, original_code):
  '''
  Takes as input question, original code, and code skeleton and generates an answer

  Args:
    quesstion (str): input question
    original_code (str): original project code
    code_skeleton (str): code skeleton to be filled out by user

  Returns:
    answer (str): answer generated based on code skeleton and original code
  '''

  prompt = """ <s>[INST] You are a helpful assistant whose role is to provide an answer to a user question based on the given context.\n
        Answer the question exclusively based on the provided context.\n
        Even if asked to, do not ever provide any python code or reference directly any python library, function or attribute.\n
        Instead, reply with "I am sorry, but I cannot directly provide any code solution". Focus then on explaining the concepts and providing hints to help the user understand and implement the solution themselves.\n
        Do not mention the context itself when replying.\n
        User question: {question}\n
        Context: {original_code}
        [/INST] </s>"""

  prompt = PromptTemplate(template=prompt, input_variables=["question", "original_code"])
  llm_chain = prompt | llm
  answer = llm_chain.invoke({"question":question, "original_code":original_code})

  return answer


def generate_feedback(submission, original_code):
  '''
  Takes as input question, original code, and code skeleton and generates an answer

  Args:
    quesstion (str): input question
    original_code (str): original project code
    code_skeleton (str): code skeleton to be filled out by user

  Returns:
    answer (str): answer generated based on code skeleton and original code
  '''

  prompt = """ <s>[INST] You are a helpful assistant whose role is to provide feedback to a user submission based on the original task.\n
        Given the original code, assess whether or not the user submission is correct and give the user a short and concise feedback on their submission.\n
        The user submission should fullfill the same function as the original code, even though the implementation might differ.\n
        Make sure to generate your feedback in markdown format.\n
        Pretend you are directly speaking to the user when answering.\n
        User submission: {submission}\n
        Original task: {original_code}
        [/INST] </s>"""

  prompt = PromptTemplate(template=prompt, input_variables=["submission", "original_code"])
  llm_chain = prompt | llm
  answer = llm_chain.invoke({"submission":submission, "original_code":original_code})

  return answer


def project_chosen(project):
    st.session_state.choice = True
    st.session_state.project = project


def choose_project(query):
    top3_projects = rerank_passages(query)
    st.markdown("Here are 3 projects that fit your request. Chose the one you want to work on.")
    col1, col2, col3 = st.columns(3)
    with col1:
        help0 = top3_projects[0]["description"] + "\n\n" + top3_projects[0]["structure"]
        st.button(top3_projects[0]["topic"], on_click=project_chosen, args=(top3_projects[0],), help=help0, use_container_width=True)
    with col2:
        help1 = top3_projects[1]["description"] + "\n\n" + top3_projects[1]["structure"]
        st.button(top3_projects[1]["topic"], on_click=project_chosen, args=(top3_projects[1],), help=help1, use_container_width=True)
    with col3:
        help2 = top3_projects[2]["description"] + "\n\n" + top3_projects[2]["structure"]
        st.button(top3_projects[2]["topic"], on_click=project_chosen, args=(top3_projects[2],), help=help2, use_container_width=True)


def next_task():
    st.session_state.task += 1
    st.session_state.feedback = True
    reset_chat()


with st.sidebar:
    HF_TOKEN = st.text_input(
        label="Enter your HuggingFace Access Token to start\n\nhttps://huggingface.co/settings/tokens", type="password",
        key="text_input")
    if HF_TOKEN:
        st.session_state.token = True
        llm_container = st.empty()
        with st.spinner(f"Loading LLM: {LLM_MODEL}"):
            # Define HuggingFace tokens needed to access the model and the Inference API
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
            os.environ["HF_TOKEN"] = HF_TOKEN
            # Define the LLM model through the HuggingFace Inference API and set temperature to 0.01 to ensure near-deterministic behavior
            llm = load_llm_model()
            llm_container.success(f"{LLM_MODEL} loaded successfully!")

        embed_container = st.empty()
        with st.spinner(f"Loading Embedding Model: {EMBED_MODEL}"):
            # Define embedding model
            embed_model = load_embed_model(HF_TOKEN)
            embed_container.success(f"{EMBED_MODEL} loaded successfully!")

        rerank_container = st.empty()
        with st.spinner(f"Loading Reranker Model: {RERANK_MODEL}"):
            # Define embedding model
            rerank_model = load_rerank_model()
            rerank_container.success(f"{RERANK_MODEL} loaded successfully!")

        collection_container = st.empty()
        with st.spinner(f"Loading Collection: {COLLECTION_NAME}"):
            try:
                collection = client.get_collection(name=COLLECTION_NAME)
                collection_container.success(f"{COLLECTION_NAME} collection loaded successfully!")
            except ValueError:
                collection_container.warning(f"{COLLECTION_NAME} collection does not exist and will be created first...")
                descriptions, metas = parse_corpus()
                embeddings = generate_embeddings(descriptions)
                index_passages(descriptions, embeddings, metas)
                collection_container.success(f"{COLLECTION_NAME} collection loaded successfully!")

        st.success("CodeCompanion is ready to Chat!")

if st.session_state.token:
    # column1, column2 = st.columns([0.8, 0.2])
    if not st.session_state.choice:
        with st.chat_message("assistant"):
            st.markdown("What kind of project you are looking for?")
        prompt = st.chat_input("I am looking for...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner(f"Retrieving relevant projects..."):
                    choose_project(prompt)
    else:
        if st.session_state.task < int(st.session_state.project["steps"]) and st.session_state.feedback:
            with st.chat_message("assistant"):
                st.markdown("This is your current task:")
                submission = st_ace(st.session_state.project[f"task_{st.session_state.task}"])
            if st.button("Submit task", help="Rember to apply the code before submitting"):
                st.session_state.feedback = False
                with st.chat_message("assistant"):
                    with st.spinner("Generating feedback..."):
                        feedback = generate_feedback(submission, st.session_state.project[f"code_{st.session_state.task}"])
                        st.session_state.messages.append({"role": "assistant", "content": feedback})
                    st.markdown(feedback)
                st.button("Start next task", on_click=next_task)
            if question := st.chat_input("How can I help?"):
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = generate_chat_answer(question, st.session_state.project[f"code_{st.session_state.task}"])
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.markdown(answer)
        elif st.session_state.task == int(st.session_state.project["steps"]):
            st.subheader("You have completed your project. Thank you for using CodeCompanion!")