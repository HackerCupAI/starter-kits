import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import logging
import logging.handlers

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

ENABLE_LOGGING = False
working_dir = "/home/autogen/autogen/app"
timestamp = time.time()
logname = f"{working_dir}/logs/{timestamp}_simple_log.log"
logger = None
if ENABLE_LOGGING:
    # Create and configure logger
    logging.basicConfig(filename=logname,
                        format='%(asctime)s %(message)s',
                        filemode='w')

    logger = logging.getLogger("test-orchestration-simple")

    logger.setLevel(logging.DEBUG)


class SimpleSelfInspectingCoder(ConversableAgent):
    def __init__(self, n_iters = 3, **kwargs):
        """
        Initializes a SimpleSelfInspectingCoder instance.

        This agent facilitates solving coding tasks through a collaborative effort among its child agents: commander, coder, and critics.

        Parameters:
            - n_iters (int, optional): The number of "improvement" iterations to run. Defaults to 3.
            - **kwargs: keyword arguments for the parent AssistantAgent.
        """
        super().__init__(**kwargs)
        self.register_reply([Agent, None], reply_func=SimpleSelfInspectingCoder._reply_user, position=0)
        self._n_iters = n_iters

    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            print(error_msg)
            if ENABLE_LOGGING:
                logger.error(error_msg)  # noqa: F821
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
            system_message="Help me run the code, and tell other agents it is in the <txt code.txt> file location.",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={"last_n_messages": 3, "work_dir": working_dir, "use_docker": True},
            llm_config=self.llm_config,
        )

        ### Critics: LMM-based agent that provides comments and feedback on the generated image.
        critics = MultimodalConversableAgent(
            name="Critics",
            system_message="""Criticize the proposed code solution to the given problem. 
            Find bugs and issues that might not otherwise thrown interpreter errors.
            Your output must be in the format below:
            Reasoning: 
            Suggestion: 
            Verdict: 

            Be judicious and write down your step-by-step evaluation of the code under "Reasoning", 
            then propose how this code can be modified so that it meets the guidelines in "Suggestion". 
            Your suggestion should be succinct. Do not include the modified code, just describe how the code should be changed. 
            Finally, "Verdict" should be either NO_ISSUES if you think the code is verifiably successful at solving the 
            original task or FAIL otherwise if there are suggestions.""",
            llm_config=VISION_CONFIG,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config = {"use_docker":True}
        )

        ### Coder: writes code
        coder = AssistantAgent(
            name="Coder",
            llm_config=self.llm_config,
            code_execution_config={"use_docker": True}
        )

        coder.update_system_message(
            "# filename: code.txt" + coder.system_message
            + "ALWAYS save the current code in `code.txt` file. Tell other agents it is in the code.txt file location. Execute code using sample input provided."
        )

        # Data flow begins
        commander.initiate_chat(coder, message=user_question)
        
        if ENABLE_LOGGING:
            coder_repsonse = commander._oai_messages[coder][-1]["content"]
            logger.info( f"sender=coder_repsonse to commander: {coder_repsonse}")
                

        for i in range(self._n_iters):
            commander.send(
                message=f" The original prompt for the problem was: "  
                + user_question
                +" Improve <txt {os.path.join(working_dir, 'code.txt')}> ",
                recipient=critics,
                request_reply=True,
            )

            feedback = commander._oai_messages[critics][-1]["content"]
            if ENABLE_LOGGING:
                  logger.info(f"sender=critics to commander: {feedback}")
            if feedback.find("NO_ISSUES") >= 0:
                break
            commander.send(
                message="Here is the feedback to your code. Please improve! Save the result to `code.txt`\n"
                + feedback,
                recipient=coder,
                request_reply=True,
            )
            if ENABLE_LOGGING:
                coder_response = commander._oai_messages[coder][-1]["content"]
                logger.info( f"sender=coder_repsonse to commander: {coder_response}")

        return True, os.path.join(working_dir, "code.txt")
   



creator = SimpleSelfInspectingCoder(name="Self Inspecting Coder", llm_config=GPT4_CONFIG)

user_proxy = UserProxyAgent(
    name="User", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config={
        "use_docker": True
    },  
)

user_proxy.initiate_chat(
    creator,
    message="""
Plot a figure by using the data from:
https://raw.githubusercontent.com/vega/vega/main/docs/data/seattle-weather.csv

Show both temperature high and low. Save the chart to weather.png
""",
)

