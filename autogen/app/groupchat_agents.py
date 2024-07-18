import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import logging

from config.config import VISION_CONFIG, GPT4_CONFIG, WORKING_DIR, DEFAULT_TIMEOUT
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
logname = f"{WORKING_DIR}/logs/group_agent_{timestamp}.log"


ENABLE_LOGGING = True
# Create and configure logger
logger = None
if ENABLE_LOGGING:
    logging.basicConfig(
        filename=logname, format="%(asctime)s %(message)s", filemode="w"
    )

    logger = logging.getLogger("self-inspecting-coder")
    logger.setLevel(logging.INFO)


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


def save_results(solution_function: str, inputs: str) -> None:
    try:
        code_obj = compile(solution_function, "<string>", "exec")
        import types
        fn = types.FunctionType(code_obj.co_consts[0], globals())
        results = fn(inputs)
        with open(f"{WORKING_DIR}\generated_out.txt", "w") as handle:
            handle.write(results)
        with open(f"{WORKING_DIR}\generated_code.txt", 'w') as handle:
            handle.write(solution_function)
    except Exception as e:
        print( f"error excuting solution: {str(e)}")

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
    def __init__(self,  images = {}, inputs={}, input_samples={}, output_samples={}, **kwargs):
        """
        Initializes a SelfInspectingCoder instance.

        This agent facilitates solving coding tasks through a collaborative effort among its child agents: project_manager, coder, and critics.

        Parameters:
            - **kwargs: keyword arguments for the parent AssistantAgent.
        """
        super().__init__(**kwargs)
        self.register_reply(
            [Agent, None], reply_func=SelfInspectingCoder._reply_user, position=0
        )
        self._input_samples = input_samples
        self._output_samples = output_samples        
        self._images = images
        self._inputs = inputs

    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            if ENABLE_LOGGING:
                logger.error(error_msg)
            raise AssertionError(error_msg)
        
        if messages is None:
            messages = self._oai_messages[sender]

        user_question = messages[-1]["content"]

        vision_capability = VisionCapability(lmm_config=VISION_CONFIG)

        ### Define the agents
        project_manager = AssistantAgent(
            name="Project_Manager",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            system_message="""Project Manager. You are facilitating a team problem solving which first starts with Image_explainer; then decide to call the next agents until the problem is solved. This may need a feedback loop to work.
            Tell all other agent the code is in <txt generated_code.txt>
            Once the code passes tests, call Writer to write the code into <txt  generated_code.txt> before exiting  """,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config,
            code_execution_config=False,
        )
        vision_capability = VisionCapability(lmm_config=VISION_CONFIG)

        ## use image_agent to get captions for image tags
        image_agent = AssistantAgent(
            name="Image_explainer",
            max_consecutive_auto_reply=2,
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
            system_message="""Logic Critic. Criticize the proposed solution from Soulution_architech to the given problem. 
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
            system_message=f"""Coder. Create code in Python based on discussions from Solution Architect and Logic Critic .
            Make sure your code follows strictly the input and output format specification from Problem_analyst.
            Must write code to parse inputs in the format specified by Problem_analyst, which might involve type coverstions to get input into right format.
            Must generate output in the correct format specified Problem_analyst.  
            Make you your code contains tests using sample input data: {self._input_samples} and sample output data: {self._output_samples}.
            Write your code into <txt generated_code.txt> and tell the team. Do not excute the code,
            """,
        ) 
        coder.update_system_message(
            "# filename: generated_code.txt" + coder.system_message
            + "ALWAYS save the current code in `generated_code.txt` file. Tell other agents it is in the generated_code.txt file location. Execute code using sample input provided."
        )

        tester = autogen.AssistantAgent(
            name="Tester",
            llm_config=GPT4_CONFIG,
            system_message=f"""Tester. Test the code in <txt generated_code.txt> using sample inputs:{self._input_samples} and sample output:{self._output_samples}
            In addtion, come up with more test cases to cover corner cases and extreme values.
            Once it passes the test, please run generated code on longer inputs {self._inputs} and save the results
            """,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={"work_dir": WORKING_DIR, "use_docker": True, "timeout":DEFAULT_TIMEOUT},
        )

        tester.register_for_llm(
            name='test_code',
            description="Test generated code by running sample inputs and compare results with sample outputs",
        )(solution_validator)

        tester.register_for_llm(
            name="save_results",
            description="Run generated code on longer inputs and save the output to file",
        )(save_results)

        executor = autogen.AssistantAgent(
            name="CodeExecutor",
            llm_config=False,
            code_execution_config={"executor": InputExecutor()},
            is_termination_msg=lambda msg: "TERMINATE"
            in msg.get("content", "").strip().upper(),
        )

        register_function(
            save_results,
            caller=tester,
            executor=executor,
            name="save_results",
            description="Run generated code on longer inputs and save the output to file",
        )

        register_function(
            save_results,
            caller=tester,
            executor=executor,
            name="solution_validator",
            description="Test generated code by running sample inputs and compare results with sample outputs",
        )

        # 
        ## ref: https://microsoft.github.io/autogen/docs/notebooks/agentchat_groupchat_research
        groupchat = autogen.GroupChat(
            agents=[
                project_manager,
                image_agent,
                problem_analyst,
                solution_architect, 
                logic_critic,
                coder,
                tester,
                executor,
            ],
            messages=[],
            max_round=12,
        )

        group_chat_manager = autogen.GroupChatManager( groupchat=groupchat, llm_config=GPT4_CONFIG)
        vision_capability.add_to_agent(group_chat_manager)

        # Data flow begins
        project_manager.initiate_chat(group_chat_manager, message=f"{user_question}")
        
        if ENABLE_LOGGING:
            coder = project_manager._oai_messages[coder][-1]["content"]
            logger.info( f"sender=coder_repsonse to project_manager: {coder_repsonse}")
        
        return True, os.path.join(WORKING_DIR, "agent_code.txt")


 