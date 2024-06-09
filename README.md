# CodeCompanion
## This project is based of the course project for AI4SI, a class at ETHZ Zurich.

### What is CodeCompanion

CodeCompanion is an AI coding assistant that proposes coding projects to users, breaks them down into single tasks, and supports the user in solving each task. By grounding state-of-the-art Natural Language Processing (NLP) and Generative AI models in existing projects, CodeCompanion addresses the lack of personalized guidance with minimal likelihood of hallucinating and the lack of accurate guidance that fails to adapt to changing user needs. CodeCompanion enhances the coding learning process by providing relevant, personalized learning experiences. It can contribute to reducing the digital divide and empowering more individuals with the coding skills necessary 

### Instructions to run CodeCompanion

1. Create a [HuggingFace account and generate access token](https://huggingface.co/docs/hub/en/security-tokens) to be used once CodeCompanion runs on your browser
2. Clone this repository to your local machine
3. Open terminal and run the commands below:
    - Run `cd <directory/to/cloned/repo>`
    - Run `python3 -m venv .venv`
    - Run `pip3 install -r requirements.txt`
    - Run `streamlit run CodeCompanion.py`
4. Input email if requested to access Streamlit
5. CodeCompanion should be running in your browser. Request a project idea to CodeCompanion. Happy Coding!
