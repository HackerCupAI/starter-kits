import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import logging

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from config.config import VISION_CONFIG, GPT4_CONFIG, WORKING_DIR
from utils.utils import get_problemset

import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.code_utils import content_str

timestamp = time.time()
logname = f"{WORKING_DIR}/logs/{timestamp}-log.log"


ENABLE_LOGGING = False
# Create and configure logger
logger = None
if ENABLE_LOGGING:
    logging.basicConfig(filename=logname,
                        format='%(asctime)s %(message)s',
                        filemode='w')

    logger = logging.getLogger("test-orchestration")

    logger.setLevel(logging.DEBUG)



### STEPS:
## PLANNER: tools/ stategies/ chat with critic agent, 
## satsify initial prompt and strategy 
## once solution go to coder if needed
## CODER: compile code, test with inputs /output and if not, go back to planning agent 

## limit loop--> max 2 iterations 
## sampling --> multiple test solutions 

## make sure works with sample of input

class SelfInspectingCoder(ConversableAgent):
    def __init__(self, n_iters = 3, input = {}, output= {}, **kwargs):
        """
        Initializes a SelfInspectingCoder instance.

        This agent facilitates solving coding tasks through a collaborative effort among its child agents: commander, coder, and critics.

        Parameters:
            - n_iters (int, optional): The number of "improvement" iterations to run. Defaults to 3.
            - **kwargs: keyword arguments for the parent AssistantAgent.
        """
        super().__init__(**kwargs)
        self.register_reply([Agent, None], reply_func=SelfInspectingCoder._reply_user, position=0)
        self._n_iters = n_iters
        self._input_samples = input["contents"]
        self._output_samples = output["contents"]


 
    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            # logging.error({"message": error_msg})  # noqa: F821
            print(f"error: {error_msg}")
            if ENABLE_LOGGING:
                logger.error(error_msg)
            raise AssertionError(error_msg)
        if messages is None:
            messages = self._oai_messages[sender]

        user_question = messages[-1]["content"]

        vision_capability = VisionCapability(lmm_config=VISION_CONFIG)

        ### Define the agents
        commander = AssistantAgent(
            name="Project_Manager",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            system_message="Project Manager. Oversees the entire project, ensures that the team stays on track. Facilitates communication among team members and end communication when solution found. Must first engage Image_explainer to caption images. ",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
             code_execution_config={"last_n_messages": 3, "work_dir": WORKING_DIR, "use_docker": True},
            llm_config=self.llm_config,
        )
        vision_capability = VisionCapability(lmm_config=VISION_CONFIG)

        ## use image_agent to get captions for image tags
        image_agent =  autogen.AssistantAgent(
            name="Image_explainer",
            max_consecutive_auto_reply=10,
            llm_config=VISION_CONFIG,
            # system_message="Describe images with factual information",
            system_message="Describe any images in img tags. Remove img tags and replace with corresponding captions. Captions should have as much detail as possible. Return original input content with caption modifications.",
            code_execution_config=False,
        )

        problem_analyst = AssistantAgent(
            name="Problem_analyst",
            system_message="""Problem Definer. Focuses on understanding the problem statement.Rephrase the problem statement into a precise goal. Reduce the problem into factual statements. List Assumptions.
              Identifying inputs and outputs and sample data for testing. Ensures that the team has a clear understanding of the problem. Do not try to solve the problem.
            """,
            llm_config=GPT4_CONFIG,
        )
        solution_architect = AssistantAgent(
            name="Solution_architect",
            system_message="""Solution Architect. 
            Devises 2-4 mathematical conditions for solving the problem. Break down the problem into smaller tasks and outlines logic to be used without coding. You should not write code. Solving the problem might be just logic or coding. 
            """,
            llm_config=GPT4_CONFIG,
        )
      
        engineer = autogen.AssistantAgent(
            name="Engineer",
            llm_config=GPT4_CONFIG,
            system_message="""Engineer. Implements a solution based on the plan devised by the Solution Architect. It may involve applying logic to solve or coding a solution. Is there a solution without coding?
          
        """,
            code_execution_config={"use_docker": True, "work_dir": WORKING_DIR}
        )
        optimizer = autogen.AssistantAgent(
            name="Optimizer",
            llm_config=GPT4_CONFIG,
            system_message="""Optimizer. Analyzes the solution from engineer for efficiency, simplicity, generalizablity, and suggests improvements. 
            If there is no optimizations to make, reply NO_OPTIMIZATIONS. 
        """,
            code_execution_config={"use_docker": True, "work_dir": WORKING_DIR}
        )

    
        executor = autogen.UserProxyAgent(
            name="Tester",
            system_message="Tester. Execute the code written by the engineer and report the result. Always use the test cases provided in original problem. If it passes tests, write the code solution in code.txt",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": True, "work_dir": WORKING_DIR  },  
        )

        ## ref: https://microsoft.github.io/autogen/docs/notebooks/agentchat_groupchat_research
        groupchat = autogen.GroupChat(
            agents=[commander, image_agent, problem_analyst, solution_architect, engineer, optimizer, executor], messages=[], max_round=12
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=GPT4_CONFIG)
        group_chat_manager = autogen.GroupChatManager(
            groupchat=groupchat, llm_config=GPT4_CONFIG
        )
        vision_capability.add_to_agent(group_chat_manager)


        commander.initiate_chat(manager, message=f"{user_question}")
        

            
        return True, os.path.join(WORKING_DIR, "code.txt")
   
# Get problem/input/output from assets/practice_problem
(problem_statement, input, output, images) = get_problemset()
problem_content = problem_statement["contents"]
creator = SelfInspectingCoder(name="Self Inspecting Coder~", input=input, output=output,  llm_config=GPT4_CONFIG)

user_proxy = autogen.UserProxyAgent(
    name="User", 
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,  
    code_execution_config={"use_docker": True, "work_dir": WORKING_DIR}
)




# Describe the following images in the following text: 
user_proxy.initiate_chat(
    recipient=creator,
    message=f"""{problem_content}"""
)


