from groupchat_agents import SelfInspectingCoder
from config.config import  GPT4_CONFIG
from utils.utils import get_problemset
import argparse
from autogen import UserProxyAgent
from time import time 



def main():

    parser = argparse.ArgumentParser(description="Solve Hackercup coding challenge.")
    parser.add_argument('data_dir', help="Dir containing all data needed for the challenge")
    
    args = parser.parse_args()
    data_dir = args.data_dir


    #NOTE, this is the data dir "/home/autogen/autogen/app/assets/nim_sum_dim_sum"
    (problem_statement, inputs, outputs, input_samples, output_samples, images) = get_problemset(data_dir)  
    
    problem_content = problem_statement["contents"]
    
    # Groupchat agents 
    creator = SelfInspectingCoder(
        name="Self Inspecting Coder",  
        llm_config=GPT4_CONFIG, 
        inputs=inputs,
        input_samples=input_samples,
        output_samples=output_samples,
        images=images
    )

    user_proxy = UserProxyAgent(
        name="User_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config= None #{"use_docker": True, "work_dir": WORKING_DIR },
        )

  
    start_time = time()
    user_proxy.initiate_chat(recipient=creator, message=f"""solve the problem:\n{problem_content}""")
    print(f"Agent exited the excusion, took: {(time()-start_time)/60} min ")



if __name__ == "__main__":
    main()


