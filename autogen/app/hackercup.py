from groupchat_agents import SelfInspectingCoder
from config.config import  GPT4_CONFIG, WORKING_DIR,DEFAULT_TIMEOUT
from utils.utils import get_problemset
import argparse
import autogen
from time import time 



def main():

    parser = argparse.ArgumentParser(description="Solve Hackercup coding challenge.")
    parser.add_argument('data_dir', help="Dir containing all data needed for the challenge")
    
    args = parser.parse_args()
    data_dir = args.data_dir


    #NOTE, this is the data dir "/home/autogen/autogen/app/assets/nim_sum_dim_sum"
    # Get problem/input/output from assets/practice_problem
    (problem_statement, inputs, outputs, input_samples, output_samples, images) = get_problemset(data_dir)  
    
    problem_content = problem_statement["contents"]
    image_contents = images['contents']
    input_contents = inputs['contents']
    input_sample_contents = input_samples['contents']
    output_sample_contents = output_samples['contents']

    
    creator = SelfInspectingCoder(
        name="Self Inspecting Coder",  
        llm_config=GPT4_CONFIG, 
        inputs=input_contents,
        input_samples=input_sample_contents,
        output_samples=output_sample_contents,
        images=image_contents
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={"use_docker": True, "work_dir": WORKING_DIR, 'timeout':DEFAULT_TIMEOUT },)

    try:
        start_time = time()
        user_proxy.initiate_chat(recipient=creator, message=f"""Solve the problem:\n{problem_content}\nSave the genterated code to generated_code.txt""")
        print(f"Agent exited the excusion, took: {(time()-start_time)/60} min ")
    except Exception as e:
        print(f'{e}')


if __name__ == "__main__":
    main()


