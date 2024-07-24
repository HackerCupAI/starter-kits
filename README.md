## Starter kits

We've provided a few starter kits for you to modify and copy paste. Additional info about each kit is available in their respective folders

We've provided all historical Hacker Cup problems in a single dataset on HuggingFace: https://huggingface.co/datasets/hackercupai/hackercup

## Using open models
1. [sample_data_solver](sample_data_solver), which evaluates CodeLlama, or any other causal language model on the HuggingFace model hub, with the sample input-output pairs only passed in the prompt. No other information (including problem statement) is used.
2. [finetuning](finetuning), which finetunes a pretrained causal language model, e.g., CodeLlama, on the Hackercup dataset, treated as a language modeling task. Evaluate off-the-shelf or finetuned model by generating n code solutions for each Hackercup problem, choosing the best (according to sample test case correctness), and evaluating on full test case input-output pairs.

# Using closed models
1. [autogen](autogen/) which is a programming framework for agentic AI https://microsoft.github.io/autogen/
2. [SWE-agent](SWE-agent/) starter kit which solves leetcode style problems https://princeton-nlp.github.io/SWE-agent/usage/coding_challenges/
3. [langchain](langchain/) starter kit which is a general framework to create LLM applications

While the Hacker Cup competition is proceeding, you will download all problems in a round in a single compressed folder and it's up to you to uncompress it and feed it to your model. Once your model produces the answer which should include both the generated code and an output.txt, you'll submit that code just like a human would on the hacker cup website.

As we're getting ready to announce winners, we'll be going in order from the top ranking winners to the bottom ranking ones
1. For the open track until we find enough reproducible answers
2. For the closed track until we find enough models that we can query via an API
