from groupchat_agents import SelfInspectingCoder
from config.config import  GPT4_CONFIG
from utils.utils import get_problemset
import argparse
from autogen import UserProxyAgent
import time 
import asyncio

min_wait_between_retry = 1
max_wait_between_retry = 10


async def init_conversation(problem_data, problem_id):
    problem_content = problem_data["problem"]
    input_samples = {"contents" : problem_data["sample_input"], "location": problem_data["sample_input_file"] }  
    output_samples = {"contents" : problem_data["sample_output"], "location": problem_data["sample_output_file"] }  
    inputs = {"content" : problem_data["input"], "location": problem_data["input_file"]}
    images = problem_data["images"]
    
    creator = SelfInspectingCoder(
        name="Self Inspecting Coder",  
        llm_config=GPT4_CONFIG, 
        inputs=inputs,
        input_samples=input_samples,
        output_samples=output_samples,
        images=images,
        problem_id=problem_id
    )

    user_proxy = UserProxyAgent(
        name="User_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config= None #{"use_docker": True, "work_dir": WORKING_DIR },
    )


    return  user_proxy.initiate_chat(recipient=creator, message=f"""solve the problem:\n{problem_content}""")



async def agent_coroutine(sem, problem_data, key):
    async with sem:
        try:
            result = await init_conversation(problem_data, key)  
            return result

        except Exception as e:
            print(e)
           

async def run_agents():
    sem = asyncio.Semaphore(10)
    
    start_time = time.time()
    all_tasks = []

    parser = argparse.ArgumentParser(description="Solve Hackercup coding challenge.")
    parser.add_argument('data_dir', help="Dir containing all data needed for the challenge")
    
    args = parser.parse_args()
    data_dir = args.data_dir


    #NOTE, this is the data dir "/home/autogen/autogen/app/assets/"
    data = get_problemset(data_dir)

    all_tasks = []
    for problem_id in list(data.keys()):
        problem_data = data[problem_id]
        task = asyncio.create_task(agent_coroutine(sem, problem_data, problem_id))
        all_tasks.append(task)
        
    result = await asyncio.gather(*all_tasks)
        
    end_time = time.time()
    print(f"api time: {end_time - start_time} seconds")
    return result




if __name__ == "__main__":
    asyncio.run(run_agents())


