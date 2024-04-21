# AI4SI-2024

Todo:
- Create a structure for the chunking so that the LLM has the correct context depending on the stage of the user interaction:
  - Ideally, at the beginning the LLM should consider the entire repo as context in order to identify the best matching project
  - After a project has been selected, only that project should be considered as context. Ideally, the user would receive by the LLM the skeleton of the code that they can change and ask answeres on.
  - Implement chatbot option that remembers past responses
