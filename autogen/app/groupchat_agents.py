import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import logging

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


ENABLE_LOGGING = True
# Create and configure logger
logger = None
if ENABLE_LOGGING:
    logging.basicConfig(
        filename=logname, format="%(asctime)s %(message)s", filemode="w"
    )

    logger = logging.getLogger("test-orchestration")

    logger.setLevel(logging.DEBUG)


def solution_validator(solution_function: str, inputs: str, outputs:str) -> str:
    try:
        # d = {}
        # exec(solution_function) in d
        code_obj = compile(solution_function, "<string>", "exec")
        import types

        fn = types.FunctionType(code_obj.co_consts[0], globals())

        results = fn(inputs, outputs)
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
    def __init__(self,  n_iters=3, images = {}, inputs={}, input_samples={}, output_samples={}, **kwargs):
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
        self._input_samples = input_samples
        self._output_samples = output_samples
        self._images = images
        self._inputs = inputs

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
            system_message="""Project Manager. You are facilitating a team problem solving which first starts with Image_explainer, Problem_analyst, Solution_architect, Logic_critics, Coder, Code_crtics, Excutor. 
            Tell all other agent the code is in <txt agent_code.txt> and solution is in <txt agent_output.txt>
            This may need a feedback loop to work. """,
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
            system_message="""Image_explainer. Describe any images in img tags. Remove img tags and replace with corresponding captions. Captions should have as much detail as possible. Return original input content with caption modifications.
            """,
            code_execution_config=False,
        )

        problem_analyst = AssistantAgent(
            name="Problem_analyst",
            system_message=f"""Problem Definer. 
            Focuses on understanding the problem statement.
            Rephrase the problem statement into a precise goal. Reduce the problem into factual statements. List Assumptions, constraints and simplify win conditions. 
            Analyze elements on the problem. 
            Give a list of Key strategy considerations.
            Provide clearn input and output format specifications. 
            Use input data, {self._input_samples} and output data {self._output_samples} as your examples.
            Ensures that the team has a clear understanding of the problem. Do not try to solve the problem.
            """,
            llm_config=GPT4_CONFIG,
        )
        solution_architect = AssistantAgent(
            name="Solution_architect",
            system_message="""Solution Architect. Devises 2-4 approaches for solving the problem. 
            Breaks down the problem into smaller tasks and outlines the logic to be used without coding. 
            There might be tricks and red herrings to throw you off; stick to the facts and reason through the problem. 
            Pay special care to the constraints of the problem. Would your solution blow up in runtime or memory? 
            You should not write code. Solve the problem with logic and reasoning.
            """,
            llm_config=GPT4_CONFIG,
        )
        logic_critic = AssistantAgent(
            name="Logic_critic",
            system_message="""Criticize the proposed solution from Soulution_architech to the given problem. 
            Be judicious and walk through the sample inputs and outputs, and corner cases and extreme values not covered in the sample.
            There might be tricks and red herrings to throw you off; stick to the facts and reason through the problem.
            Consider runtime and memeory usage before suggesting a solution 
            Your output must be in the format below:
            Reasoning: 
            Suggestion: 
            Verdict: 

            Write down your step-by-step evaluation of the solution under "Reasoning", 
            then propose how the solution might be improved in "Suggestion". 
            Your suggestion should be succinct. Do not write code. 
            Finally, "Verdict" should be either NO_ISSUES if you think the solution is good, or FAIL otherwise if there are suggestions.""",
            llm_config=GPT4_CONFIG,
        )

        coder = autogen.AssistantAgent(
            name="Coder",
            llm_config=GPT4_CONFIG,
            system_message=f"""Coder. Create code in Python based on discussions from Solution Architect and Logic_Critic .
            Make sure your code follows strictly the input and output format specification from Problem_analyst.
            Must write code to parse inputs in the format specified by Problem_analyst, which might involve type coverstions to get input into right format.
            Must generate output in the correct format specified Problem_analyst.  
            Make you your code contains tests using sample input data: {self._input_samples} and sample output data: {self._output_samples}.
            Write your code into generated_code.txt and tell the team. Do not excute the code,
            """,
        ) 
        coder.update_system_message(
            "# filename: generated_code.txt" + coder.system_message
            + "ALWAYS save the current code in `generated_code.txt` file. Tell other agents it is in the generated_code.txt file location." 
        )  
        code_critic = autogen.AssistantAgent(
            name="Code_critic",
            llm_config=GPT4_CONFIG,
            system_message=f"""Criticize the proposed code solution from Coder 
            Find bugs and issues that might not otherwise thrown interpreter errors.
            Your output must be in the format below:
            Reasoning: 
            Suggestion: 
            Verdict: 

            Be judicious and write down your step-by-step evaluation of the code under "Reasoning",
            use sample inputs:{self._input_samples} and sample outputs: {self._output_samples},
            and come up with more test cases to cover corner cases and extreme values. Let other agent know your addional test cases.
            Then propose how this code can be modified so that it meets the guidelines in "Suggestion".
            Do NOT fix the code yourself. 
            Your suggestion should be succinct. Do not include the modified code, just describe how the code should be changed. 
            Finally, "Verdict" should be either NO_ISSUES if you think the code is verifiably successful at solving the 
            original task or FAIL otherwise if there are suggestions.""",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config = {"use_docker":True}
        )

        code_excutor =autogen.AssistantAgent(
            name="Code_excutor",
            llm_config=GPT4_CONFIG,
            system_message=f"""Code_excutor. Once Code_critic approves the code with NO_ISSUES certic, run the code on {self._inputs} and generate output in agent_output.txt
            When this is complete, declare task complete and exit""",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config = {"use_docker":True}
        )


        # 
        ## ref: https://microsoft.github.io/autogen/docs/notebooks/agentchat_groupchat_research
        groupchat = autogen.GroupChat(
            agents=[
                commander,
                image_agent,
                problem_analyst,
                solution_architect, 
                logic_critic,
                coder,
                code_critic,
                code_excutor,
            ],
            messages=[],
            max_round=12,
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=GPT4_CONFIG)
        group_chat_manager = autogen.GroupChatManager(
            groupchat=groupchat, llm_config=GPT4_CONFIG
        )
        vision_capability.add_to_agent(group_chat_manager)

        commander.initiate_chat(manager, message=f"{user_question}")

        return True, os.path.join(WORKING_DIR, "agent_code.txt"), os.path.join(WORKING_DIR, "agent_output.txt")


