This folder contains the implementation of a RAG agent to solve the Hacker Cup problems using LLMs.
It includes scripts for downloading and preprocessing the datasets and generating solutions using a Retrieval Model.

The RAG agent is based on a combination of a retriever and a generator model.
The retriever is used to retrieve similar historical problems and solutions from
the [codecontests](https://huggingface.co/datasets/deepmind/code_contests) dataset and prompt an LLM with few-shot
examples to generate solutions for the current problem.

You can learn more about the approach in this youtube video:

<a target="_blank" href="https://www.youtube.com/watch?v=cObBj2UpWK8">
<img src="https://img.youtube.com/vi/cObBj2UpWK8/0.jpg" width="600" height="450">
</a>

## Contents

1. `demo.ipynb`: this notebook contains a full walkthrough of the RAG agent and how to use it to solve Hacker Cup
   problems.
2. `retriever.py`: this script contains the implementation of the retriever we used.
3. `agent.py`: this script contains the implementation of three different agents we used to solve the problems.
4. `utils.py`: utility functions used in retrieving and generating solutions.
5. `requirements.txt`: list of required packages to run the code.


