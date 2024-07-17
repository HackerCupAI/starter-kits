from groupchat_agents import SelfInspectingCoder
from config.config import  GPT4_CONFIG, WORKING_DIR
from utils.utils import get_problemset
import argparse
import autogen
from time import time 



def main():

    parser = argparse.ArgumentParser(description="Solve Hackercup coding challenge.")
    parser.add_argument('data_dir', help="Dir containing all data needed for the challenge")
    
    args = parser.parse_args()
    data_dir = args.data_dir

    start_time = time()
    #NOTE, this is the data dir "/home/autogen/autogen/app/assets/nim_sum_dim_sum"
    # Get problem/input/output from assets/practice_problem
    (problem_statement, inputs, outputs, input_samples, output_samples, images) = get_problemset(data_dir)  
    
    problem_content = problem_statement["contents"]
    image_contents = images['contents']
    input_contents = inputs['contents']
    input_sample_contents = input_samples['contents']
    output_sample_contents = output_samples['contents']

    
    creator = SelfInspectingCoder(
        name="Self Inspecting Coder~",  
        llm_config=GPT4_CONFIG, 
        inputs=input_contents,
        input_samples=input_sample_contents,
        output_samples=output_sample_contents,
        images=image_contents
    )

    user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config={"use_docker": True, "work_dir": WORKING_DIR},)
    user_proxy.initiate_chat(recipient=creator, message=f"""{problem_content}""")
    print(f"Agent exited the excusion, took: {(time()-start_time)/60} sec ")


if __name__ == "__main__":
    main()


