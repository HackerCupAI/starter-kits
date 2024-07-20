# Running the code

`python main-hackercup.py`

## Editing the code

### Changing the problem

To change the problem that the code runs on, you must replace the files in the `assets/practice_problem` folder with the correct files. Make sure to look at the `utils/utils.py` file for information on how the problem statement gets pulled.

Once you have replaced the files, be sure to edit `problem_statement_message` in the `main-hackercup.py` file and also change the code at the bottom of the file for reading in the inputs (the test case format may be different) and generating the outputs.

### Changing the agent

The agent logic is defined in `hackercup_graph.py` - if you want to modify the agent logic you should modify that.

### Improving the agent

There are various ideas for improving the graph:

1. Add more logic to the algorithm design aspect
2. Introduce more nodes and edges for the agent to plan their solution
3. Improve the system prompts
4. Add few shot examples of code solving algoirthmic problems
etc.