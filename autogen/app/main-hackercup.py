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
from autogen import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    UserProxyAgent,
    register_function,
)
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen.agentchat.contrib.multimodal_conversable_agent import (
    MultimodalConversableAgent,
)
from autogen.coding import (
    CodeBlock,
    CodeExecutor,
    CodeExtractor,
    CodeResult,
    MarkdownCodeExtractor,
)
from autogen.code_utils import content_str

timestamp = time.time()
logname = f"{WORKING_DIR}/logs/{timestamp}-log.log"


ENABLE_LOGGING = False
# Create and configure logger
logger = None
if ENABLE_LOGGING:
    logging.basicConfig(
        filename=logname, format="%(asctime)s %(message)s", filemode="w"
    )

    logger = logging.getLogger("test-orchestration")

    logger.setLevel(logging.DEBUG)

# Get problem/input/output from assets/practice_problem
(problem_statement, input, output, images) = get_problemset()
problem_content = problem_statement["contents"]


def solution_validator(solution_function: str) -> str:
    try:
        # d = {}
        # exec(solution_function) in d
        code_obj = compile(solution_function, "<string>", "exec")
        import types

        fn = types.FunctionType(code_obj.co_consts[0], globals())
        # cases = input["contents"][1:].split("\n")
        # cases = [case.split(',') for case ]
        # expected = output["contents"].split("\n")
        results = fn(input["contents"], output["contents"])
        return (
            "All cases passed"
            if len(results) == 0
            else f"Failed for cases: {results[0:5]}"
        )
    except Exception as e:
        return f"Error executing solution: {str(e)}"


class InputExecutor(CodeExecutor):

    @property
    def code_extractor(self) -> CodeExtractor:
        # Extact code from markdown blocks.
        return MarkdownCodeExtractor()


### STEPS:
## PLANNER: tools/ stategies/ chat with critic agent,
## satsify initial prompt and strategy
## once solution go to coder if needed
## CODER: compile code, test with inputs /output and if not, go back to planning agent

## limit loop--> max 2 iterations
## sampling --> multiple test solutions

## make sure works with sample of input


class SelfInspectingCoder(ConversableAgent):
    def __init__(self, n_iters=3, input={}, output={}, **kwargs):
        """
        Initializes a SelfInspectingCoder instance.

        This agent facilitates solving coding tasks through a collaborative effort among its child agents: commander, coder, and critics.

        Parameters:
            - n_iters (int, optional): The number of "improvement" iterations to run. Defaults to 3.
            - **kwargs: keyword arguments for the parent AssistantAgent.
        """
        super().__init__(**kwargs)
        self.register_reply(
            [Agent, None], reply_func=SelfInspectingCoder._reply_user, position=0
        )
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
            system_message="Project Manager. You are facilitating a team problem solving which first starts with Image_explainer, Input_parser, Problem_analyst, Solution_architect, Engineer, Tester. This may need a feedback loop to work   ",
            is_termination_msg=lambda x: x.get("content", "")
            .rstrip()
            .endswith("TERMINATE"),
            code_execution_config={
                "last_n_messages": 3,
                "work_dir": WORKING_DIR,
                "use_docker": True,
            },
            llm_config=self.llm_config,
        )
        vision_capability = VisionCapability(lmm_config=VISION_CONFIG)

        ## use image_agent to get captions for image tags
        image_agent = AssistantAgent(
            name="Image_explainer",
            max_consecutive_auto_reply=10,
            llm_config=VISION_CONFIG,
            # system_message="Describe images with factual information",
            system_message="Describe any images in img tags. Remove img tags and replace with corresponding captions. Captions should have as much detail as possible. Return original input content with caption modifications.",
            code_execution_config=False,
        )
        input_parser = AssistantAgent(
            name="Input_parser",
            system_message="""Input Parser. Given the input data, {self._input_samples} and output data {self._output_samples}
            take the first 50 cases and put into an array in  format  [R, C,A,B, <Yes or  No>] .
            put this in file in test_sample.py. 
            """,
            llm_config=GPT4_CONFIG,
        )

        interviewee = AssistantAgent(
            name="Intervewee",
            system_message=f"""Interviewee. You are interviewing for a software company. You need to apply logic and coding skills to answer the problem.
            Ask questions to the interviewer and make sure you understand the problem fully. Find some strategies that we can analyze the data to see if we can solve the problem. Do not provide test cases. Just provide a function that can be tested.""",
            llm_config=GPT4_CONFIG,
            code_execution_config={"use_docker": True, "work_dir": WORKING_DIR},
        )

        interviewer = AssistantAgent(
            name="Interviewer",
            system_message=f"""Interviewer. You are interviewing a candidate for a software company. The answer to the problem is if R > C Alice has garenteed win else C >= R Bob wins. 
            Do not tell the Interviewee answer or providing any code or psuedocode.  You can ask them questions about reasoning. double check all solutions with Tester.
            """,
            llm_config=GPT4_CONFIG,
        )

        problem_analyst = AssistantAgent(
            name="Problem_analyst",
            system_message="""Problem Definer. 
            Focuses on understanding the problem statement.
            Rephrase the problem statement into a precise goal. Reduce the problem into factual statements. List Assumptions, constraints and simplify win conditions. Analyze elements on the problem. 
            Give a list of Key strategy considerations.
            Ensures that the team has a clear understanding of the problem. Do not try to solve the problem.
            """,
            llm_config=GPT4_CONFIG,
        )
        solution_architect = AssistantAgent(
            name="Solution_architect",
            system_message="""Solution Architect.
            Devises 2-4 approaches for solving the problem. 
            Break down the problem into smaller tasks and outlines logic to be used without coding. 
            You should not write code. Solving the problem might be just logic or coding.
            """,
            llm_config=GPT4_CONFIG,
        )
        qa_agent = AssistantAgent(
            name="QA",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            system_message="""
                QA. Your job is to validate that the code produced by the engineer works as expected and solves the original logic puzzle. 
                If it doesn't, reply with "FAILED" and provide the failing test cases.
                If it does, respond with "TERMINATE" to end the conversation.
            """,
            is_termination_msg=lambda x: x.get("content", "")
            .rstrip()
            .endswith("TERMINATE"),
            llm_config=GPT4_CONFIG,
        )

        engineer = autogen.AssistantAgent(
            name="Engineer",
            llm_config=GPT4_CONFIG,
            system_message="""Engineer. Implements a solution based on strategies given by Solution Architect. Assume format for input in array with values [R, C,A,B].  If code does not work use feedback from team until it works.

        """,
            code_execution_config={"use_docker": True, "work_dir": WORKING_DIR},
        )
        engineer.register_for_llm(
            name="solution_validator",
            description="Takes a logic puzzle solution function, and validates it against a set of inputs and outputs.",
        )(solution_validator)
        # optimizer = autogen.AssistantAgent(
        #     name="Critic",
        #     llm_config=GPT4_CONFIG,
        #     system_message="""Critic. Analyzes the solution from engineer for efficiency, simplicity, generalizablity, and suggests improvements.
        #     If there is no optimizations to make, reply NO_OPTIMIZATIONS.
        # """,
        #     code_execution_config={"use_docker": True, "work_dir": WORKING_DIR},
        # )

        executor = autogen.AssistantAgent(
            name="CodeExecutor",
            llm_config=False,
            code_execution_config={"executor": InputExecutor()},
            is_termination_msg=lambda msg: "TERMINATE"
            in msg.get("content", "").strip().upper(),
        )

        register_function(
            solution_validator,
            caller=qa_agent,
            executor=executor,
            name="solution_validator",
            description="Takes a logic puzzle solution function, and validates it against a set of inputs and outputs.",
        )

        ## ref: https://microsoft.github.io/autogen/docs/notebooks/agentchat_groupchat_research
        groupchat = autogen.GroupChat(
            agents=[
                commander,
                image_agent,
                input_parser,
                problem_analyst,
                solution_architect,
                engineer,
                executor,
                qa_agent,
            ],
            messages=[],
            max_round=12,
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=GPT4_CONFIG)
        group_chat_manager = autogen.GroupChatManager(
            groupchat=groupchat, llm_config=GPT4_CONFIG
        )
        vision_capability.add_to_agent(group_chat_manager)

        q = user_question.replace("Nim Sum ", "")

        commander.initiate_chat(manager, message=f"{q}")

        return True, os.path.join(WORKING_DIR, "code.txt")


# Get problem/input/output from assets/practice_problem

creator = SelfInspectingCoder(
    name="Self Inspecting Coder~", input=input, output=output, llm_config=GPT4_CONFIG
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config={"use_docker": True, "work_dir": WORKING_DIR},
)


# Describe the following images in the following text:
user_proxy.initiate_chat(recipient=creator, message=f"""{problem_content}""")
