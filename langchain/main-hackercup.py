from langchain_core.messages import HumanMessage
import base64
from langchain_openai import ChatOpenAI

from hackercup_graph import get_graph_for_problem
from utils.utils import get_problemset

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

'''
In order to run this code on a problem other than the Dim Sum Delivery problem, please edit the code below

You will need to change the contents of the assets/practice_problem directory in order for it to work
'''
(problem_statement, input) = get_problemset()
problem_content = problem_statement["contents"]

# Make sure to incldue any images from the problem statement in the assets folder
image_1 = encode_image('./assets/practice_problem/852013469652032.jpg')
image_2 = encode_image('./assets/practice_problem/842253013944047.jpg')

# Change this message if you select a different problem than Dim Sum Delivery
problem_statement_message = HumanMessage(content=[
            {"type": "text", "text": f"Can you please write code to solve the following problem? {problem_content}"},
            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_1}"}},
            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_2}"}}
])

# Get the graph
graph = get_graph_for_problem(ChatOpenAI(temperature=0, model="gpt-4o"), max_iterations = 3, flag ='reflect')

# Invoke graph, starting with 0 iterations
final_ans = graph.invoke(input={"messages":[problem_statement_message]})['generation']

'''
See if the final answer is correct by checking the generated code against the test cases:

This code will need to be changed if you use a problem other than the Dim Sum Delivery problem
'''

# Get test cases
input_lines = input["contents"].split("\n")[1:]
inputs = []
for i in range(len(input_lines)):
    # NOTE: Not all problems have same test case format - make sure you append the right inputs (you may need to go through multiple lines!)
    inputs.append([int(x) for x in input_lines[i].split(" ")])

answers = []
exec(final_ans.imports + "\n" + final_ans.code)
for input in inputs:
    # Make sure to check final_ans.code to see what function it is returning - in this case it is nim_sum_dim_sum
    answers.append(nim_sum_dim_sum(*input))

with open('output.txt', 'w') as file:
    for i in range(len(answers)):
        file.write(f"Case #{i+1}: "+answers[i]+"\n")