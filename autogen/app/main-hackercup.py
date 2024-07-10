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
from config.config import VISION_CONFIG, GPT4_CONFIG

import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.code_utils import content_str
working_dir = "/home/autogen/autogen/app"
timestamp = time.time()
logname = f"{working_dir}/logs/{timestamp}-log.log"


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
    def __init__(self, n_iters = 3, **kwargs):
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

        ### Define the agents
        
        ### Commander: interacts with users, runs code, and coordinates the flow between the coder and critics.
        commander = AssistantAgent(
            name="Commander",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            system_message="Interact with the planner to discuss the plan. Plan execution needs to be approved by this Commander",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
             code_execution_config={"last_n_messages": 3, "work_dir": working_dir, "use_docker": True},
            llm_config=self.llm_config,
        )

        ## use image_agent to get captions for image tags
        image_agent = MultimodalConversableAgent(
            name="Image_explainer",
            max_consecutive_auto_reply=10,
            llm_config=VISION_CONFIG,
            system_message="Describe any images in img tags. Remove img tags and replace with corresponding captions. Captions should have as much detail as possible. Return original input content with caption modifications. Remove any warnings related to unable to load images.",
            code_execution_config=False,
        )

    

        planner = AssistantAgent(
            name="Planner",
            system_message="""Planner. 
            Rephrase the problem statement into a precise goal.
            Reduce the problem into factual statements. 
            Find any information that can be deduced from this information. 
            Walk through 2-4 hypothetical situations. Suggest at least two distinct plans to solve the problem. Be creative! This plan should not involve coding but a strategy.
            Revise the plan based on feedback from commander and critic, until critic approval.
            Identify what resources are needed. This may involve an Engineer who can write code and/or a Enigmatologist who doesn't write code but is really good at solving logic puzzles.
            Explain the plan first. Be clear which step is performed by the Engineer, and which step is performed by a Enigmatologist. Include a plan for test cases.
            """,
            llm_config=GPT4_CONFIG,
        )

        plan_critic = autogen.AssistantAgent(
            name="Plan_critic",
            system_message="""
            Critic. Double check the plan. Provide constructive critisim if necessary. Ensure plan is logical, and feasible. Ensure plan is as simple as possible. 
            Find issues that might occur during execution. Remember we do not need a solution but a strategy to solve. You should not suggest a specific strategy.
            Your output must be in the format below:
            Reasoning: 
            Suggestion: 
            Verdict: 

            Be judicious and write down your step-by-step evaluation of the code under "Reasoning", 
            then propose how this code can be modified so that it meets the guidelines in "Suggestion". 
            Your suggestion should be succinct. Do not include the modified code, just describe how the code should be changed. 
            Finally, "Verdict" should be either NO_ISSUES if you think the code is verifiably successful at solving the 
            original task or FAIL otherwise if there are suggestions.
            """,
            llm_config=GPT4_CONFIG,
        )

        engineer = autogen.AssistantAgent(
            name="Engineer",
            llm_config=GPT4_CONFIG,
            system_message="""Engineer. You follow an approved plan. You write python code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
        Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
        If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
        """,
            code_execution_config={"use_docker": True, "work_dir": working_dir}
        )
        enigmatologist = autogen.AssistantAgent(
            name="Enigmatologist",
            llm_config=GPT4_CONFIG,
            system_message="""Enigmatologist. You follow an approved plan. You are able to solve logic puzzles You don't write code.""",
        )
        executor = autogen.UserProxyAgent(
            name="Executor",
            system_message="Executor. Execute the code written by the engineer and report the result. Always use the test cases provided in the plan.",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": True, "work_dir": working_dir  },  
        )

        ## ref: https://microsoft.github.io/autogen/docs/notebooks/agentchat_groupchat_research
        groupchat = autogen.GroupChat(
            agents=[commander, planner, plan_critic, engineer, enigmatologist, executor], messages=[], max_round=12
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=GPT4_CONFIG)

        ## Get Image Captions
        image_prompt = "Add image captions to the following text:"
        commander.send(message=f"{image_prompt} {user_question}",  recipient=image_agent, request_reply=True)
        image_capturing = commander._oai_messages[image_agent][-1]["content"]
        if ENABLE_LOGGING:
            logger.info( f"sender=image_agent to commander: {image_capturing}")

        augmented_prompt_w_captions = image_capturing.replace(image_prompt, "")

        commander.initiate_chat(manager, message=f"{augmented_prompt_w_captions}")
        
    


        # ### Critics: LMM-based agent that provides comments and feedback on the generated image.
        # critics = MultimodalConversableAgent(
        #     name="Critics",
        #     system_message="""Provide constructive critisim to the proposed code solution to the given problem. Restate the original problem.
        #     Does the code attempt to solve the original problem? Is there a simplier solution? Find bugs and issues that might not otherwise thrown interpreter errors.
        #     Your output must be in the format below:
        #     Reasoning: 
        #     Suggestion: 
        #     Verdict: 

        #     Be judicious and write down your step-by-step evaluation of the code under "Reasoning", 
        #     then propose how this code can be modified so that it meets the guidelines in "Suggestion". 
        #     Your suggestion should be succinct. Do not include the modified code, just describe how the code should be changed. 
        #     Finally, "Verdict" should be either NO_ISSUES if you think the code is verifiably successful at solving the 
        #     original task or FAIL otherwise if there are suggestions.""",
        #     llm_config=VISION_CONFIG,
        #     human_input_mode="NEVER",
        #     max_consecutive_auto_reply=1,
        #     code_execution_config = {"use_docker":True}
        # )

        # ### Coder: writes code
        # coder = AssistantAgent(
        #     name="Coder",
        #     llm_config=self.llm_config,
        #     code_execution_config={"use_docker": True},
        # )

        # coder.update_system_message(
        #     "# filename: code.txt" + coder.system_message
        #     + "ALWAYS save the current code in `code.txt` file. Tell other agents it is in the code.txt file location. Execute code using sample input provided."
        # 
            
                 # for i in range(self._n_iters):
        #     if ENABLE_LOGGING:
        #       logger.info(f"ITERATION: {i}")
        #     print(f"ITERATION: {i}")
        #     commander.send(
        #         message=f" The original prompt for the problem was: "  
        #         + augmented_prompt_w_captions
        #         +" Improve <txt {os.path.join(working_dir, 'code.txt')}> ",
        #         recipient=plan_critic,
        #         request_reply=True,
        #     )

        #     feedback = commander._oai_messages[plan_critic][-1]["content"]
        #     if ENABLE_LOGGING:
        #       logger.info(f"sender=plan_critic to commander: {feedback}")
        #     if feedback.find("NO_ISSUES") >= 0:
        #         break
        #     # commander.send(
        #     #     message="Here is the feedback to your code. Please improve! Save the result to `code.txt`\n"
        #     #     + feedback,
        #     #     recipient=coder,
        #     #     request_reply=True,
        #     # )
        #     # if ENABLE_LOGGING:
        #     #     coder_response = commander._oai_messages[coder][-1]["content"]
        #     #     logger.info( f"sender=coder_repsonse to commander: {coder_response}")
        
            
        return True, os.path.join(working_dir, "code.txt")
   



creator = SelfInspectingCoder(name="Self Inspecting Coder~", llm_config=GPT4_CONFIG)

user_proxy = autogen.UserProxyAgent(
    name="User", 
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,  
    code_execution_config={"use_docker": True, "work_dir": working_dir}
)

user_proxy.initiate_chat(
    creator,
    message="""
Solve the following problem:
Nim Sum Dim Sum, a bustling local dumpling restaurant, has two game theory-loving servers named, you guessed it, Alice and Bob. Its dining area can be represented as a two-dimensional grid of R rows (numbered 1..R from top to bottom) by C columns (numbered 1..C from left to right).
Currently, both of them are standing at coordinates (1,1) where there is a big cart of dim sum. Their job is to work together to push the cart to a customer at coordinates (R,C). To make the job more interesting, they've turned it into a game.
Alice and Bob will take turns pushing the cart. On Alice's turn, the cart must be moved between 1 and A units down. On Bob's turn, the cart must be moved between 1 and B units to the right. The cart may not be moved out of the grid. If the cart is already at row R on Alice's turn or column C on Bob's turn, then that person loses their turn.
The "winner" is the person to ultimately move the cart to (R,C) and thus get all the recognition from the customer. Alice pushes first. Does she have a guaranteed winning strategy?

##Constraints
1≤T≤500 
2≤R,C≤10^9 
1≤𝐴<𝑅
1≤𝐵<𝐶
##Input Format
Input begins with an integer T, the number of test cases. Each case will contain one line with four space-separated integers, R, C, A, and B.
##Output Format
For the ith test case, print "Case #i: " followed by "YES" if Alice has a guaranteed winning strategy, or "NO" otherwise.
##Sample Explanation
The first case is depicted below, with Alice's moves in red and Bob's in blue. Alice moves down, and Bob moves right to win immediately. There is no other valid sequence of moves, so Alice has no guaranteed winning strategy.
<img https://scontent-sea1-1.xx.fbcdn.net/v/t39.32972-6/381216378_842253017277380_357307080060896787_n.jpg?_nc_cat=111&ccb=1-7&_nc_sid=771dbb&_nc_ohc=KpDNsOpFHmwQ7kNvgH8vFmq&_nc_ht=scontent-sea1-1.xx&oh=00_AYAp-j_LPuMBaw1P6DOBB6ekShr5-Yh6XBa1pheD-VaR6w&oe=669260BC />

The second case is depicted below. One possible guaranteed winning strategy is if Alice moves 3 units down, then Bob can only move 1 unit, and finally Alice can win with 1 unit.
<img https://scontent-sea1-1.xx.fbcdn.net/v/t39.32972-6/381363162_852013472985365_7715015420796665512_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=771dbb&_nc_ohc=L-4JYOGMILEQ7kNvgHodzMp&_nc_ht=scontent-sea1-1.xx&oh=00_AYAUVvvUiUVqy2_f2ikr4pk7YI0jGUi4BgyWAiE4rPkZAw&oe=66924729 />

Sample Input
3
2 2 1 1
5 2 3 1
4 4 3 3
Sample Output
Case #1: NO
Case #2: YES
Case #3: NO

Assuming Bob and Alice will use the most optimal solutions respectively, are there conditions in which alice is guaranteed to win and that can be determined before the game even begins?
""",
)

#ref: https://www.facebook.com/codingcompetitions/hacker-cup/2023/practice-round/problems/B
