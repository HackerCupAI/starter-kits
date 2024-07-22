from langchain_core.messages import HumanMessage
import base64
from langchain_openai import ChatOpenAI

from hackercup_graph import get_graph_for_problem
from utils.utils import get_problemset
import argparse
import os

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Solve Hackercup coding challenge.")
    parser.add_argument('data_dir', help="Dir containing all data needed for the challenge")
    
    args = parser.parse_args()
    data_dir = args.data_dir
    problem_statement, input, input_samples, output_samples = get_problemset(data_dir)
    problem_content = problem_statement["contents"]

    # Make sure to incldue any images from the problem statement in the assets folder
    images = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                images.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{encode_image(os.path.join(root, file))}"}})

    # Change this message if you select a different problem than Dim Sum Delivery
    problem_statement_message = HumanMessage(content=[
                {"type": "text", "text": f"Can you please write code to solve the following problem? {problem_content}"}]
                #+images
                )

    # Get the graph
    graph = get_graph_for_problem(ChatOpenAI(temperature=0, model="gpt-4o"), max_iterations = 3, flag ='reflect')
    
    # Invoke graph, starting with 0 iterations
    final_ans = graph.invoke(input={"messages":[problem_statement_message],"input_samples":input_samples,"output_samples":output_samples})['generation']

    '''
    #See if the final answer is correct by checking the generated code against the test cases:

    #This code will need to be changed if you use a problem other than the Dim Sum Delivery problem
    '''

    # Get test cases
    input_lines = input["location"].split("\n")[1:-1]
    inputs = []
    for i in range(len(input_lines)):
        # NOTE: Not all problems have same test case format - make sure you append the right inputs (you may need to go through multiple lines!)
        inputs.append([int(x) for x in input_lines[i].split(" ")])

    answers = []
    existing_functions_before_exec = set(globals().keys())
    exec(final_ans.imports + "\n" + final_ans.code)
    existing_functions_after_exec = set(globals().keys())
    # Make sure it is the correct function (length longer than 6 to avoid new variables in code)
    new_function = [l for l in list(existing_functions_after_exec - existing_functions_before_exec) if l not in ['existing_functions_before_exec', 'Tuple'] and len(l) > 6][0]
    for input in inputs:
        # Make sure to check final_ans.code to see what function it is returning - in this case it is nim_sum_dim_sum
        answers.append(globals()[new_function](*input))

    with open('output.txt', 'w') as file:
        for i in range(len(answers)):
            file.write(f"Case #{i+1}: "+answers[i]+"\n")
    