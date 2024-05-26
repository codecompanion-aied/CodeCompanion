# Import necessary libraries
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

COLLECTION_NAME = "CodeCompanion" # Name of the Chroma collection to be loaded or created
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-l" # Embedding model to be loaded through HuggingFace Embeddings Inference API
RERANK_MODEL = "BAAI/bge-reranker-large" # Reranker model to be loaded locally
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1" # LLM to be loaded through HuggingFace Inference API
TOP_K_RERANK = 50 # Number of most relevant k passages to retrieve before reranking
TOP_K = 3  # Number of most relevant k passages to retrieve after reranking


def reset_chat():
    '''
    Resets the session chat history as an empty list
    '''
    st.session_state.messages = []


# Initialize the session ID with a Universally Unique IDentifier (here version 4) and initialize the session cache as an empty dictionary
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

# Instantiate a persistent Chroma client to save to disk
client = chromadb.PersistentClient()

# Initialize the session chat history as an empty list
if "messages" not in st.session_state:
    reset_chat()

# Initialize the session HuggingFace token validation as False since the user has yet to pass the HuggingFace token
if "token" not in st.session_state:
    st.session_state.token = False

# Initialize the session project choice validation to False since the user has yet to choose a project to work on
if "choice" not in st.session_state:
    st.session_state.choice = False

# Initialize the session project as None since the user has yet to choose a project to work on
if "project" not in st.session_state:
    st.session_state.project = None

# Initialize the current session task number as zero since the user has yet to complete a task
if "task" not in st.session_state:
    st.session_state.task = 0

# Initialize the session feedback validation as True since the user has yet to approve the feedback
if "feedback" not in st.session_state:
    st.session_state.feedback = True

# Initialize the session user submission as None since the user has yet to submit a task
if "submission" not in st.session_state:
    st.session_state.submission = None

# Display the app title
st.title("CodeCompanion by Team CC #AI4Impact2024")


@st.cache_resource(show_spinner=False) # Function saved in cache after first call to prevent the app from loading the LLM at every run
def load_llm_model():
    '''
    Define the llm through the HuggingFace Inference API with max token generation length of 4096
    and temperature of 0.01 to ensure near-deterministic behavior

    Returns
        llm: LLM used by CodeCompanion
    '''

    llm = HuggingFaceEndpoint(repo_id=LLM_MODEL, max_length=4096, temperature=0.01)

    return llm


@st.cache_resource(show_spinner=False) # Function saved in cache after first call to prevent the app from loading the embedding model at every run
def load_embed_model(token):
    '''
    Define the embedding model through the HuggingFace Embeddings Inference API

    Returns
        embed_model: Embedding model used by CodeCompanion
    '''

    embed_model = HuggingFaceInferenceAPIEmbeddings(api_key=token, model_name=EMBED_MODEL)

    return embed_model


@st.cache_resource(show_spinner=False) # Function saved in cache after first call to prevent the app from loading the reranker model at every run
def load_rerank_model():
    '''
    Define the reranker model through FlagReranker with 16-bit floating point for faster performance (according to BAAI documentation)

    Returns
        rerank_model: Reranker model used by CodeCompanion
    '''

    rerank_model = FlagReranker(RERANK_MODEL, use_fp16=True)

    return rerank_model


def parse_corpus():
    '''
    Parses the csv CodeCompanion.csv containing a separate project in each row
    Returns a list of project descriptions and a list of project metadata

    Returns
        project_descriptions (list): list of project descriptions
        project_meta (list): list of project metadata
    '''

    # Initialize descriptions and meta lists
    project_descriptions = []
    project_meta = []
    # Open CodeCompanion.csv
    with open("CodeCompanion.csv") as f:
        # Define the csv reader
        data = csv.reader(f, delimiter=",")
        # Skip the header row
        idx = 0
        for row in data:
            if idx == 0:
                idx = 1
                continue
            # Save the combined first 4 cells in each row as project description (includes Domain, Topic, Description, and Structure)
            project_description = ' '.join(row[0:4])
            # Add the project description to the dedicated list
            project_descriptions.append(project_description)
            # Save the project topic, description, structure and number of steps in the metadata dictionary
            meta = {"topic": row[1], "description": row[2], "structure": row[3], "steps": row[4]}
            # Save each task code and task skeleton in the metadata dictionary
            for i in range(int(row[4])):
                meta[f"code_{i}"] = row[5 + 2 * i]
            for i in range(int(row[4])):
                meta[f"task_{i}"] = row[6 + 2 * i]
            # Add the project metadata to the dedicated list
            project_meta.append(meta)

    # Print the number of project descriptions and meta to ensure they match
    print("# descriptions: ", len(project_descriptions))
    print("# metadata: ", len(project_meta))

    return project_descriptions, project_meta


def generate_embeddings(passages):
    '''
    Takes a list of passages and generates embeddings

    Args:
      passages (list): list of passages

    Returns:
      passage_embeddings (list): list of embeddings
    '''

    # Generate the passage embeddings with the respective embedding model
    passages_embeddings = embed_model.embed_documents(passages)

    # Print the number of embeddings and ensure the dimensionality matches the expected output of the embedding model
    print("# embeddings: ", len(passages_embeddings))
    print("Dimension of first embedding: ", len(passages_embeddings[0]))

    return passages_embeddings


def index_passages(passages, embeddings, metadatas):
    '''
    Takes as input passages, embeddings, and metadata and puts in vector store
    Saves to disk

    Args:
        passages (list): list of passages
        embeddings (list): list of embeddings
        metadatas (list): list of metadata
    '''

    # Create the Chroma collection using cosine similarity
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # Add the passages, embeddings and metadata to the collection
    collection.add(
        ids=[str(i) for i in list(range(len(embeddings)))],
        documents=passages,
        embeddings=embeddings,
        metadatas=metadatas
    )


def retrieve_passages(query, top_k):
    '''
    Takes as input a query, finds the top k passages and returns them

    Args:
        query (str): query
        top_k (int): number of most relevant k passages to be returned

    Returns:
        passages (list): k most relevant passages
        metadatas (list): metadata corresponding to k most relevant passages
    '''

    # Load the Chroma collection
    collection = client.get_collection(name=COLLECTION_NAME)

    # Embed the query to search for relevant passages
    query_embed = embed_model.embed_query(query)
    # Print the dimension of the query embedding to check if it matches the document embedding dimension
    print("Dimension of query embedding: ", len(query_embed))

    # Retrieve the k most relevant passages and the corresponding metadata
    results = collection.query(
    query_embeddings=query_embed,
    n_results=top_k,
    include = ["documents", "metadatas"]
    )

    # Assume that only one query is passed
    passages = results['documents'][0]
    metadatas = results['metadatas'][0]

    return passages, metadatas


def rerank_passages(query):
    '''
    Takes as input a query, finds the top n passages and corresponding metadata, reranks them, and returns the top k metadata

    Args:
    query (str): query

    Returns
    reranked_metadata (list): metadata corresponding to k most relevant, reranked passages
    '''

    # Retrieve the n most relevant passages and corresponding metadata
    passages, metadatas = retrieve_passages(query, TOP_K_RERANK)

    # Compute the relevance scores through the reranker model
    scores = rerank_model.compute_score([[query, passage] for passage in passages])
    # Format the scores as percentages
    scores = [round(100 * score, 1) for score in scores]
    # Print the number of scores to check that it matches the number of passages
    print("# scores: ", len(scores))

    # Initialize the list of reranked metadata
    reranked_metadatas = []
    # Add the metadata and corresponding relevance score to a dictionary withing the reranked metadata list
    for metadata, score in zip(metadatas, scores):
        reranked_metadatas.append({"metadata": metadata, "score": score})

    # Sort the list of reranked metadata based on the relevance scores in descending order
    reranked_metadatas = sorted(reranked_metadatas, key=lambda x: x["score"], reverse=True)
    # Retrieve only the top k entries of the reranked metadata list
    reranked_metadatas = reranked_metadatas[0:TOP_K]

    # Remove the scores from the list to only keep the metadata
    reranked_metadata = []
    for meta in reranked_metadatas:
        reranked_metadata.append(meta["metadata"])

    return reranked_metadata


def generate_chat_answer(question, original_code):
    '''
    Takes as input question and the task's original code, and generates an answer

    Args:
    question (str): input question
    original_code (str): original task code

    Returns:
    answer (str): answer to the question generated based on the original code
    '''

    # Define the prompt to be passed to the LLM before generating the answer
    prompt = """ <s>[INST] You are a helpful assistant whose role is to provide an answer to a user question based on the given context.\n
        Answer the question exclusively based on the provided context.\n
        Even if asked to, do not ever provide any python code or reference directly any python library, function or attribute.\n
        Instead, reply with "I am sorry, but I cannot directly provide any code solution". Focus then on explaining the concepts and providing hints to help the user understand and implement the solution themselves.\n
        Do not mention the context itself when replying.\n
        User question: {question}\n
        Context: {original_code}
        [/INST] </s>"""

    # Add the prompt to the prompt template and define the question and the original code as input variables
    prompt = PromptTemplate(template=prompt, input_variables=["question", "original_code"])
    # Create the LLM chain
    llm_chain = prompt | llm
    # Generate the answer to the question by using the original code as context
    answer = llm_chain.invoke({"question":question, "original_code":original_code})

    return answer


def generate_feedback(submission, original_code):
    '''
    Takes as input the user task submission and the task's original code, and generates a feedback

    Args:
    submission (str): user task submission
    original_code (str): original task code

    Returns:
    feedback (str): feedback to the user submission generated based on the original code
    '''

    # Define the prompt to be passed to the LLM before generating the feedback
    prompt = """ <s>[INST] You are a helpful assistant whose role is to provide feedback to a user submission based on the original task.\n
        Given the original code, assess whether or not the user submission is correct and give the user a short and concise feedback on their submission.\n
        The user submission should fullfill the same function as the original code, even though the implementation might differ.\n
        Make sure to generate your feedback in markdown format.\n
        Pretend you are directly speaking to the user when answering.\n
        User submission: {submission}\n
        Original task: {original_code}
        [/INST] </s>"""

    # Add the prompt to the prompt template and define the user submission and the original code as input variables
    prompt = PromptTemplate(template=prompt, input_variables=["submission", "original_code"])
    # Create the LLM chain
    llm_chain = prompt | llm
    # Generate the feedback to the user submission by using the original code as context
    feedback = llm_chain.invoke({"submission":submission, "original_code":original_code})

    return feedback


def project_chosen(project):
    '''
    Takes as input the chosen project, sets it as session project, and validates the session project choice

    Args:
        project (dict): chosen project
    '''

    st.session_state.project = project
    st.session_state.choice = True


def choose_project(query):
    '''
    Takes as input a query, retrieves and displays the 3 most relevant projects, and lets the user chose one

    Args:
        query (str): query
    '''

    # Retrieves the top 3 projects for the query
    top3_projects = rerank_passages(query)
    # Outputs the assistant's message
    st.markdown("Here are 3 projects that fit your request. Chose the one you want to work on.")
    # Displays the retrieved projects as buttons with their topic on them in 3 columns.
    # When hovered over, they display also the description and structure of the project.
    # By clicking the corresponding button the user can choose the project.
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
    '''
    Increases the current session task counter, validates the session feedback and resets the session chat history
    '''

    st.session_state.task += 1
    st.session_state.feedback = True
    reset_chat()


# Display content in the sidebar
with st.sidebar:
    # Ask the user to input their HuggingFace access token (the token will be hidden upon submission)
    HF_TOKEN = st.text_input(
        label="Enter your HuggingFace Access Token to start\n\nhttps://huggingface.co/settings/tokens", type="password",
        key="text_input")
    if HF_TOKEN:
        # If the token has been submitted, validate the session token input
        st.session_state.token = True
        # Initialize an empty container to display the successful loading of the LLM
        llm_container = st.empty()
        with st.spinner(f"Loading LLM: {LLM_MODEL}"):
            # Define HuggingFace tokens needed to access the LLM and the Inference API as environment variables
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
            os.environ["HF_TOKEN"] = HF_TOKEN
            # Load the LLM model
            llm = load_llm_model()
            # Display the successful loading of the LLM
            llm_container.success(f"{LLM_MODEL} loaded successfully!")
        # Initialize an empty container to display the successful loading of the embedding model
        embed_container = st.empty()
        with st.spinner(f"Loading Embedding Model: {EMBED_MODEL}"):
            # Load the embedding model
            embed_model = load_embed_model(HF_TOKEN)
            # Display the successful loading of the embedding model
            embed_container.success(f"{EMBED_MODEL} loaded successfully!")
        # Initialize an empty container to display the successful loading of the reranker model
        rerank_container = st.empty()
        with st.spinner(f"Loading Reranker Model: {RERANK_MODEL}"):
            # Load the reranker model
            rerank_model = load_rerank_model()
            # Display the successful loading of the reranker model
            rerank_container.success(f"{RERANK_MODEL} loaded successfully!")
        # Initialize an empty container to display the successful loading of the Chroma collection
        collection_container = st.empty()
        with st.spinner(f"Loading Collection: {COLLECTION_NAME}"):
            # Try to load the Chroma collection if already existing
            try:
                collection = client.get_collection(name=COLLECTION_NAME)
            # If not already existing, display a warning, parse the corpus, embed the passages and generate the collection
            except ValueError:
                collection_container.warning(f"{COLLECTION_NAME} collection does not exist and will be created first...")
                descriptions, metas = parse_corpus()
                embeddings = generate_embeddings(descriptions)
                index_passages(descriptions, embeddings, metas)
            # Display the successful loading of the Chroma collection
            collection_container.success(f"{COLLECTION_NAME} collection loaded successfully!")
        # Display to indicate that the user can now chat with CodeCompanion
        st.success("CodeCompanion is ready to Chat!")

# Check if the user has inputted the HuggingFace token
if st.session_state.token:
    if not st.session_state.choice:
        # If the user has not yet chosen a project, ask them to do so
        with st.chat_message("assistant"):
            st.markdown("What kind of project you are looking for?")
        prompt = st.chat_input("I am looking for...")
        if prompt:
            # If the user has inputted what project they are looking for, display their prompt
            with st.chat_message("user"):
                st.markdown(prompt)
            # Search 3 projects related to the user prompt and display them in the assistant's reply
            with st.chat_message("assistant"):
                with st.spinner(f"Retrieving relevant projects..."):
                    choose_project(prompt)
    else:
        # If the user has chosen a project, display the project's tasks one at a time
        if st.session_state.task < int(st.session_state.project["steps"]) and st.session_state.feedback:
            # Display the current task in the assistant's message through the ace code editor
            with st.chat_message("assistant"):
                st.markdown("This is your current task:")
                submission = st_ace(st.session_state.project[f"task_{st.session_state.task}"])
            if st.button("Submit task", help="Rember to apply the code before submitting"):
                # If the user clicks the button to submit the current task, set the session feedback validation to False
                st.session_state.feedback = False
                # Display the assistant's feedback based on the user submission
                with st.chat_message("assistant"):
                    with st.spinner("Generating feedback..."):
                        feedback = generate_feedback(submission, st.session_state.project[f"code_{st.session_state.task}"])
                        st.session_state.messages.append({"role": "assistant", "content": feedback})
                    st.markdown(feedback)
                # Let the user pass to the next task if they are satisfied with the received feedback
                st.button("Start next task", on_click=next_task)
            if question := st.chat_input("How can I help?"):
                # If the user asks a question while working on the current task, display the current chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                # Add the user question to the chat history
                st.session_state.messages.append({"role": "user", "content": question})
                # Display the user question
                with st.chat_message("user"):
                    st.markdown(question)
                # Generate and display the assistant's response to the user question based on the task's code
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = generate_chat_answer(question, st.session_state.project[f"code_{st.session_state.task}"])
                        # Add the assistant's answer to the chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.markdown(answer)
        elif st.session_state.task == int(st.session_state.project["steps"]):
            # If the user has completed all tasks, inform the user
            st.subheader("You have completed your project. Thank you for using CodeCompanion!")
else:
    # Before the user has inputted their HuggingFace token, display a short introduction to CodeCompanion
    st.subheader("CodeCompanion is your personal coding assistant that suggests projects you can work on based on your "
                 "needs and helps you implement them step by step.\n\n"
                 "To start chatting with CodeCompanion, you will need to input your HuggingFace access token "
                 "as CodeCompanion is based on several models hosted through the HuggingFace Inference API."
                 "This makes CodeCompanion not only fast but also light to self-host and completely free to use!\n\n"
                 "After submitting your token, simply tell CodeCompanion what type of project you are looking for "
                 "and it will suggest to you three projects matching your description.\n\n"
                 "After choosing what project you want to work on, CodeCompanion will present to you sequentially each "
                 "task that makes up the project. You will be asked to complete each task to complete the project.\n\n"
                 "If you need any help while completing the tasks, simply ask CodeCompanion. He's here to help you in "
                 "your coding journey and will try its best to make you a coding wizard in no time!")