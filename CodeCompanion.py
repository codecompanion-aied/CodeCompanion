import os
import gc
import uuid
import time
import csv
import streamlit as st
import chromadb
from langchain import PromptTemplate
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint


PATH_TO_SAVE_EMBEDDINGS = ""
COLLECTION_NAME = "CodeCompanion"
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-l"
RERANK_MODEL = "BAAI/bge-reranker-large"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TOP_K = 3  # Number of most relevant k passages to retrieve
TOP_K_RERANK = 50 # Number of most relevant k passages to retrieve before reranking
COLLECTION_CREATED = True

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
session_id = st.session_state.id
client = chromadb.PersistentClient()

proj_1_descr = "Basics Password Generator This project is a simple password generator that creates random, secure passwords based on user-specified criteria. It utilizes the Python secrets module for generating cryptographically strong random numbers and the string module for creating a character set. 1. Define the Character Set: Use the string module to include all ASCII letters, digits, and punctuation.\n2. Password Generation Function: Write a function to generate a single password of specified length by randomly selecting characters from the defined set.\n3. User Interaction: Implement a loop to repeatedly ask the user for the number of passwords and their lengths, handling exceptions for non-integer inputs.\n4. Generate and Display Passwords: For the number of passwords specified, generate each and print them out."
proj_1_meta = {
    'description': 'This project is a simple password generator that creates random, secure passwords based on user-specified criteria. It utilizes the Python secrets module for generating cryptographically strong random numbers and the string module for creating a character set.',
    'structure': '1. Define the Character Set: Use the string module to include all ASCII letters, digits, and punctuation.\n2. Password Generation Function: Write a function to generate a single password of specified length by randomly selecting characters from the defined set.\n3. User Interaction: Implement a loop to repeatedly ask the user for the number of passwords and their lengths, handling exceptions for non-integer inputs.\n4. Generate and Display Passwords: For the number of passwords specified, generate each and print them out.',
    'steps': '4',
    'code_0': 'import string\n\ndef define_character_set():\n    return string.ascii_letters + string.digits + string.punctuation\n\n# Function call to see the character set\ncharacter_set = define_character_set()\nprint(character_set)',
    'code_1': "import random\n\ndef generate_password(length, character_set):\n    return ''.join(random.choice(character_set) for _ in range(length))\n\n# Sample dataset and function call\ncharacter_set = string.ascii_letters + string.digits + string.punctuation\npassword = generate_password(10, character_set)  # Generate a 10-character password\nprint(password)",
    'code_2': 'def get_user_input():\n    while True:\n        try:\n            num_pass = int(input("How many passwords do you want to generate? "))\n            password_length = int(input("Enter the length of the password(s): "))\n            return num_pass, password_length\n        except ValueError:\n            print("Please enter a valid integer.")\n\n# Function call to execute the step\nnum_pass, password_length = get_user_input()\nprint(f"Number of Passwords: {num_pass}, Length of Each Password: {password_length}")',
    'code_3': 'def generate_and_display_passwords(num_pass, password_length, character_set):\n    print("Generated passwords:")\n    for i in range(num_pass):\n        password = generate_password(password_length, character_set)\n        print(f"{i+1}. {password}")\n\n# Sample dataset and function call\ncharacter_set = string.ascii_letters + string.digits + string.punctuation\nnum_pass, password_length = 5, 12  # Generate 5 passwords, each 12 characters long\ngenerate_and_display_passwords(num_pass, password_length, character_set)',
    'task_0': 'import string\n\ndef define_character_set():\n    # This function should return a string containing all ASCII letters, digits, and punctuation\n    pass\n\n# Function call to see the character set\ncharacter_set = define_character_set()\nprint(character_set)',
    'task_1': "import random\n\ndef generate_password(length, character_set):\n    # Implement a function that returns a string consisting of 'length' randomly chosen characters from 'character_set'\n    pass\n\n# Sample dataset and function call\ncharacter_set = string.ascii_letters + string.digits + string.punctuation\npassword = generate_password(10, character_set)  # Generate a 10-character password\nprint(password)",
    'task_2': 'def get_user_input():\n    # Implement a loop that repeatedly asks for two integers: the number of passwords and the length of each password.\n    # Use try-except to handle non-integer inputs and loop until valid integers are provided.\n    pass\n\n# Function call to execute the step\nnum_pass, password_length = get_user_input()\nprint(f"Number of Passwords: {num_pass}, Length of Each Password: {password_length}")',
    'task_3': "def generate_and_display_passwords(num_pass, password_length, character_set):\n    # Implement a function that prints 'num_pass' passwords, each 'password_length' long, using the 'character_set'.\n    # Each password should be printed on a new line with an index.\n    pass\n\n# Sample dataset and function call\ncharacter_set = string.ascii_letters + string.digits + string.punctuation\nnum_pass, password_length = 5, 12  # Generate 5 passwords, each 12 characters long\ngenerate_and_display_passwords(num_pass, password_length, character_set)"
    }

proj_2_descr = "Basics Random Name Generator This project involves creating a Python script that fetches a list of names from an online source and randomly selects a name from the list. It uses the requests library to download the data from a given URL, which contains a plain text file of proper names. The random library is then used to select a name at random from this list. This could be particularly useful for applications such as generating random usernames, test data, or for any use case where random name selection is needed. 1. Import Libraries:\nImport the requests module to handle HTTP requests.\nImport the random module to enable random selection from the list.\n2. Fetch Data:\nUse the requests.get() method to download the content from a predefined URL that points to a plain text file containing proper names.\n3. Data Processing:\nProcess the fetched data by splitting the text into individual names using the split() method. This step removes any empty spaces and organizes the data into an iterable list of names.\n4. Random Name Selection:\nUtilize the random.choice() method to select and print a random name from the list of names. This function is directly used on the list obtained from the splitting operation to pick a name at random."
proj_2_meta = {
    'description': 'This project involves creating a Python script that fetches a list of names from an online source and randomly selects a name from the list. It uses the requests library to download the data from a given URL, which contains a plain text file of proper names. The random library is then used to select a name at random from this list. This could be particularly useful for applications such as generating random usernames, test data, or for any use case where random name selection is needed.',
    'structure': '1. Import Libraries:\nImport the requests module to handle HTTP requests.\nImport the random module to enable random selection from the list.\n2. Fetch Data:\nUse the requests.get() method to download the content from a predefined URL that points to a plain text file containing proper names.\n3. Data Processing:\nProcess the fetched data by splitting the text into individual names using the split() method. This step removes any empty spaces and organizes the data into an iterable list of names.\n4. Random Name Selection:\nUtilize the random.choice() method to select and print a random name from the list of names. This function is directly used on the list obtained from the splitting operation to pick a name at random.',
    'steps': '4',
    'code_0': """def import_libraries():\n    # Import the required modules for HTTP requests and random operations.\n    import requests\n    import random\n\n# Function call to import libraries\nimport_libraries()""",
    'code_1': """import requests\n\ndef fetch_data(url):\n    # Fetch content from the specified URL using the requests.get() method.\n    response = requests.get(url)\n    if response.status_code == 200:\n        return response.text\n    else:\n        return \"Failed to retrieve data\"\n\n# Function call to fetch data\nurl = 'https://svnweb.freebsd.org/csrg/share/dict/propernames?revision=61766&view=co'\nnames = fetch_data(url)\nprint(names[:500])  # Print the first 500 characters of the fetched data for demonstration""",
    'code_2': """def process_data(data):\n    # Split the data into individual names using the split() method.\n    return data.split()\n\n# Function call to process data\nindividual_words = process_data(names)\nprint(individual_words[:10])  # Print the first 10 names to verify the split""",
    'code_3': """import random\n\ndef select_random_name(names_list):\n    # Select and print a random name from the list using random.choice().\n    return random.choice(names_list)\n\n# Function call to select a random name\nrandom_name = select_random_name(individual_words)\nprint(random_name)""",
    'task_0': """def import_libraries():\n    # Import the requests module for HTTP requests and the random module for random operations.\n    pass\n\n# Function call to import libraries\nimport_libraries()""",
    'task_1': """import requests\n\ndef fetch_data(url):\n    # Use requests.get() to download data from a URL and return the text content of the response.\n    pass\n\n# Function call to fetch data\nurl = 'https://svnweb.freebsd.org/csrg/share/dict/propernames?revision=61766&view=co'\nnames = fetch_data(url)\nprint(names[:500])  # Example of checking the first 500 characters of the data""",
    'task_2': """def process_data(data):\n    # Split the input data into a list of individual names using the split() method.\n    pass\n\n# Function call to process data\nindividual_words = process_data(names)\nprint(individual_words[:10])  # Example of checking the first 10 processed items""",
    'task_3': """import random\n\ndef select_random_name(names_list):\n    # Use the random.choice() method to select a random name from the provided list.\n    pass\n\n# Function call to select a random name\nrandom_name = select_random_name(individual_words)\nprint(random_name)"""
    }

proj_3_descr = "Basics Shorten Links This project is a simple Python script that utilizes the pyshorteners library to shorten URLs. The script prompts the user to input a long URL and then uses the tinyurl service provided by pyshorteners to generate a shortened version of the URL. It's particularly useful for making lengthy URLs more manageable and easier to share, especially on platforms where character space is limited."
proj_3_meta = {
    'description': "This project is a simple Python script that utilizes the pyshorteners library to shorten URLs. The script prompts the user to input a long URL and then uses the tinyurl service provided by pyshorteners to generate a shortened version of the URL. It's particularly useful for making lengthy URLs more manageable and easier to share, especially on platforms where character space is limited.",
    'structure': "1. Import Library:\nImport the pyshorteners module, which is used to interact with various URL shortening services.\n2. Capture User Input:\nPrompt the user to enter the URL they wish to shorten. This is done using the input() function to capture user input from the command line.\n3. Shorten URL:\nInitialize a Shortener object from the pyshorteners library.\nUse the tinyurl service within pyshorteners to shorten the provided URL.\nStore the shortened URL in a variable.\n4. Display Result:\nPrint the shortened URL to the console, providing the user with the transformed link.\n5. Execution:\nThe script is designed to run interactively: once started, it will prompt for input, process that input to shorten the URL, and then display the result—all in a straightforward, linear sequence.",
    'steps': '5',
    'code_0': """def import_library():\n    # Import the pyshorteners module to access URL shortening services.\n    import pyshorteners\n\n# Function call to import library\nimport_library()""",
    'code_1': """def capture_user_input():\n    # Prompt the user to enter a URL to be shortened.\n    link = input(\"\\nEnter your link: \")\n    return link\n\n# Function call to capture user input\nuser_link = capture_user_input()""",
    'code_2': """import pyshorteners\n\ndef shorten_url(link):\n    # Create a Shortener object and use it to shorten the provided URL using the tinyurl service.\n    shortener = pyshorteners.Shortener()\n    short_url = shortener.tinyurl.short(link)\n    return short_url\n\n# Function call to shorten the URL\nshortened_url = shorten_url(user_link)""",
    'code_3': """def display_result(short_url):\n    # Print the shortened URL to the console.\n    print(\"\\nShortened link is: \" + short_url)\n\n# Function call to display the shortened URL\ndisplay_result(shortened_url)""",
    'code_4': """if __name__ == '__main__':\n    # Execute the URL shortening process interactively.\n    import_library()\n    user_link = capture_user_input()\n    shortened_url = shorten_url(user_link)\n    display_result(shortened_url)""",
    'task_0': """def import_library():\n    # Import the pyshorteners library needed for URL shortening services.\n    pass\n\n# Function call to import library\nimport_library()""",
    'task_1': """def capture_user_input():\n    # Prompt the user to enter a URL that they wish to shorten and return it.\n    pass\n\n# Function call to capture user input\nuser_link = capture_user_input()""",
    'task_2': """import pyshorteners\n\ndef shorten_url(link):\n    # Use the pyshorteners library to create a shortener object and shorten the provided URL.\n    pass\n\n# Function call to shorten the URL\nshortened_url = shorten_url(user_link)""",
    'task_3': """def display_result(short_url):\n    # Output the shortened URL to the console.\n    pass\n\n# Function call to display the shortened URL\ndisplay_result(shortened_url)""",
    'task_4': """if __name__ == '__main__':\n    # Run the full process of importing the library, capturing user input, shortening the URL, and displaying the result.\n    import_library()\n    user_link = capture_user_input()\n    shortened_url = shorten_url(user_link)\n    display_result(shortened_url)"""
    }

proj_descr = [proj_1_descr, proj_2_descr, proj_3_descr]
proj_meta = [proj_1_meta, proj_2_meta, proj_3_meta]


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


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
            meta = {"description": row[2], "structure": row[3], "steps": row[4]}
            for i in range(int(row[4])):
                meta[f"code_{i}"] = row[5 + 2 * i]
            for i in range(int(row[4])):
                meta[f"task_{i}"] = row[6 + 2 * i]
            project_meta.append(meta)

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

    return passages_embeddings


def index_passages(embeddings, meta):
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
        embeddings=embeddings,
        metadatas=meta
    )


st.header("CodeCompanion by Team CC #AI4Impact2024")
text_input = st.empty()
HF_TOKEN = text_input.text_input(label="Enter your HuggingFace Access Token to start\n\nhttps://huggingface.co/settings/tokens", type="password", key="text_input")
if HF_TOKEN:
    llm_container = st.empty()
    with st.spinner(f"Loading LLM: {LLM_MODEL}"):
        # Define HuggingFace tokens needed to access the model and the Inference API
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
        os.environ["HF_TOKEN"] = HF_TOKEN
        # Define the LLM model through the HuggingFace Inference API and set temperature to 0.01 to ensure near-deterministic behavior
        llm = HuggingFaceEndpoint(
            repo_id=LLM_MODEL, max_length=4096, temperature=0.01
        )
        llm_container.success(f"{LLM_MODEL} loaded successfully!")

    embed_container = st.empty()
    with st.spinner(f"Loading Embedding Model: {EMBED_MODEL}"):
        # Define embedding model
        embed_model = HuggingFaceInferenceAPIEmbeddings(model_name=EMBED_MODEL)
        embed_container.success(f"{EMBED_MODEL} loaded successfully!")

    rerank_container = st.empty()
    with st.spinner(f"Loading Reranker Model: {RERANK_MODEL}"):
        # Define embedding model
        rerank_model = HuggingFaceInferenceAPIEmbeddings(model_name=RERANK_MODEL)
        rerank_container.success(f"{RERANK_MODEL} loaded successfully!")

    collection_container = st.empty()
    with st.spinner(f"Loading Collection: {COLLECTION_NAME}"):
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            collection_container.success(f"{COLLECTION_NAME} collection loaded successfully!")
        except ValueError:
            collection_container.warning(f"{COLLECTION_NAME} collection does not exist and will be created first...")
            descriptions, meta = parse_corpus()
            embeddings = generate_embeddings(descriptions)
            index_passages()
            collection_container.success(f"{COLLECTION_NAME} collection loaded successfully!")

    cc_ready = st.empty()
    cc_ready.success("CodeCompanion is ready to Chat!")

    if st.session_state.text_input != "":
        time.sleep(1)
        text_input.empty()
        time.sleep(0.5)
        llm_container.empty()
        time.sleep(0.5)
        embed_container.empty()
        time.sleep(0.5)
        rerank_container.empty()
        time.sleep(1)
        cc_ready.empty()


def retrieve_passages(query, top_k):
  '''
  Takes as input a query and finds the top k passages and returns them

  Args:
    query (str): query
    top_k (int): number of most relevant passages to be returned

  Returns
    metadatas (list): metadata corresponding to k most relevant passages
  '''

  # Load the collection "legal_cases"
  collection = client.get_collection(name=COLLECTION_NAME)
  # Retrieve the top k most relevant passages, related metadata and inner product distances
  results = collection.query(
    query_texts=query,
    n_results=top_k,
    include = ["metadatas"]
  )

  # Get top passages, metadata and scores for the given query as first index, as it asssumes list of queries as input
  metadatas = results['metadatas'][0]

  return metadatas


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

  # Retrieve n most relevant passages
  top_passages, metadatas, scores = retrieve_passages(query, TOP_K_RERANK)

  # Initialize reranked_context
  reranked_context = []

  # Itarate over each top n passage and corresponding metadata
  for passage, metadata in zip(top_passages, metadatas):
    # Compute the relevance score between the query and the passage
    score = cross_encoder.predict([(query, passage)])
    # Format the score as percentage
    score = round(100 * score[0], 1)
    # Append a dict with passage, metadata and score to the reranked_context list
    reranked_context.append({"passage": passage, "metadata": metadata, "score": score})

  # Rerank the list of dicts based on each passage's relevance score
  reranked_context = sorted(reranked_context, key=lambda x: x["score"], reverse=True)
  # Consider only the top k most relevant passages
  reranked_context = reranked_context[0:TOP_K_RETRIEVE]

  # Initialize lists
  reranked_passages = []
  reranked_metadata = []
  reranked_scores = []

  # Split each value of the dicts into the corresponding lists
  for context in reranked_context:
    reranked_passages. append(context["passage"])
    reranked_metadata.append(context["metadata"])
    reranked_scores.append(context["score"])

  # Print the most relevant, reranked passage, corresponding metadata and corresponding relevance score
  print("Top reranked passage: ", reranked_passages[0])
  print("Top ranked metadata: ", reranked_metadata[0])
  print("Top reranked score: ", str(reranked_scores[0]))

  return reranked_passages, reranked_metadata, reranked_scores


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


# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if HF_TOKEN:
    if prompt := st.chat_input("What kind of project are you looking for?"):
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