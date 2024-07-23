import time
import logging
import os
from config.config import VISION_API_ENABLED, VISION_CONFIG, GPT4_CONFIG, WORKING_DIR, BASE_LOGS_DIR
from utils.utils import mkdirp

import autogen
from autogen import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent
)
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability



LOGS_DIR = f"{BASE_LOGS_DIR}/{time.strftime('%Y%m%d-%H%M%S')}/"
ENABLE_LOGGING = False

# Create and configure logger
# WARNING: autogen.runtime_logging seems to crash when using image agent (vision), maybe related to  https://github.com/microsoft/autogen/pull/3127
logger = None
if ENABLE_LOGGING:

    # Ensure the logs directory exists
    mkdirp(LOGS_DIR)

    # Logs specifically for the app
    app_log = f"{LOGS_DIR}/app.log"

    # Logs for the agents
    agents_log = f"{LOGS_DIR}/agents.log"

    logger = logging.getLogger("self-inspecting-coder")
    logger.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(app_log)
    fh.formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.mode = 'w'
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Start autogen logs 
    logging_session_id = autogen.runtime_logging.start(logger_type="file", config={"filename": agents_log})




class SelfInspectingCoder(ConversableAgent):
    def __init__(self, n_iters=3, images = {}, inputs={}, input_samples={}, output_samples={}, problem_id="", **kwargs):
        """
        Initializes a SelfInspectingCoder instance.

        This agent facilitates solving coding tasks through a collaborative effort among its child agents: 
        Project Manager, Image Explainer, Problem Analyst, Solution Architect, Logic Critic, Coder, Code Critic.

        Parameters:
            - **kwargs: keyword arguments for the parent AssistantAgent.
        """
        super().__init__(**kwargs)
        self.register_reply(
            [Agent, None], reply_func=SelfInspectingCoder._reply_user, position=0
        )
        # define number of attempts
        self._n_iters = n_iters
        # supporting data
        self._input_samples = input_samples
        self._output_samples = output_samples        
        self._images = images
        self._inputs = inputs
        self._problem_id = problem_id

    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            if ENABLE_LOGGING:
                logger.error(error_msg)
            raise AssertionError(error_msg)
        
        if messages is None:
            messages = self._oai_messages[sender]

        user_question = messages[-1]["content"]


        ### Define the agents
        project_manager = UserProxyAgent(
            name="Project_Manager",
            max_consecutive_auto_reply=10,
            system_message=f"""Project Manager. You are facilitating a team problem solving which first starts with Image_explainer; then decide to call the next agents until the problem is solved. This may need a feedback loop to work.
            Tell all other agent the code is in <txt {self._problem_id}_generated_code.txt>
            Once code passes validation or fails twice, reply with TERMINATE
            """,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            # human_input_mode="TERMINATE",
            llm_config=self.llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )

        problem_analyst = AssistantAgent(
            name="Problem_analyst",
            system_message=f"""Problem Analyst. 
            Focuses on understanding the problem statement.
            Rephrase the problem statement into a precise goal. Reduce the problem into factual statements. List Assumptions, constraints and simplify win conditions. 
            Analyze elements on the problem. 
            Give a list of Key strategy considerations.
            Provide clearn input and output format specifications. 
            Use input data, {self._input_samples['contents']} and output data {self._output_samples['contents']} as your examples.
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

        coder = AssistantAgent(
            name="Coder",
            llm_config=GPT4_CONFIG,
            system_message=f"""Coder. you are an expert in writing Python code. You must write 1 program and do all the following steps:
            1. Solve the problem using discussions with the Solution Architect and Logic Critic. 
            Ensure your code strictly follows the input and output format specifications provided by the Problem Analyst.
            The code should read inputs from a file and generate results to another file. Allow the caller to specify the locations of these two files.
            2.Test your code with input sample file: {self._input_samples['location']}, compare your output with example outputs {self._output_samples['contents']}
            3.Run your code on input file:{self._inputs['location']} and save output to {self._problem_id}_generated_output.txt
            4.Save your code into {self._problem_id}_generated_code.txt 
            5.Verify your code is saved correctly.
            Once you complete these steps, notify the group.
            """,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            code_execution_config={"work_dir": WORKING_DIR, "use_docker": True,}

        ) 

        code_critic = AssistantAgent(
            name="Code_critic",
            system_message=f"""Code Critic. Criticize the proposed code solution to the given problem. 
            Find bugs and issues that might not otherwise thrown interpreter errors.
            Your output must be in the format below:
            Reasoning: 
            Suggestion: 
            Verdict: 

            Be judicious and write down your step-by-step evaluation of the code under "Reasoning", 
            then propose how this code can be modified so that it meets the guidelines in "Suggestion". 
            Your suggestion should be succinct. Do not include the modified code, just describe how the code should be changed. 
            Finally, "Verdict" should be either NO_ISSUES if you think the code is verifiably successful at solving the 
            original task or FAIL otherwise if there are suggestions.
            If code fails tests twice,  reply with TERMINATE. 
            """,
            llm_config=GPT4_CONFIG,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            code_execution_config={"work_dir": WORKING_DIR, "use_docker": True},
        )

        ## ref: https://microsoft.github.io/autogen/docs/notebooks/agentchat_groupchat_research
        groupchat = GroupChat(
            agents=[
                project_manager,
                problem_analyst,
                solution_architect,
                logic_critic,
                coder,
                code_critic,
            ],
            messages=[],
            max_round=12,
        )

        group_chat_manager = GroupChatManager( groupchat=groupchat, llm_config=GPT4_CONFIG)
        ## if using image_agent inside group_chat_manager, use  vision_capability.add_to_agent(group_chat_manager)

        user_response_with_captions = user_question

        if not ENABLE_LOGGING and VISION_API_ENABLED:
            ## use image_agent to get captions for image tags
            image_agent = AssistantAgent(
                name="Image_explainer",
                max_consecutive_auto_reply=2,
                llm_config=VISION_CONFIG,
                system_message="""Image Explainer. Describe any images in <img > tags. Remove img tags and replace with corresponding captions. Do not remove <txt > tags.
                Captions should have as much detail as possible. Return original input content with caption modifications.
                Do not write code or analyize the problem, only explain the images.
                """,
                code_execution_config=False,
            )
            vision_capability = VisionCapability(lmm_config=VISION_CONFIG)
            vision_capability.add_to_agent(image_agent)

            project_manager.send(
                message=f"Add image captions to the following {user_question}", 
                recipient=image_agent,
                request_reply=True,
            )
            user_response_with_captions = project_manager._oai_messages[image_agent][-1]["content"]
        # Data flow begins
        attempt = 0
        while attempt < self._n_iters:
            try:
                project_manager.initiate_chat(group_chat_manager, message=f"{user_response_with_captions}")
                last_response = project_manager._oai_messages[group_chat_manager][-1]["content"]
                if last_response.find("NO_ISSUES") >= 0:
                    break
            except Exception as e:
                if ENABLE_LOGGING:
                    logger.error(e)
                print(e)
            attempt += 1 

        if ENABLE_LOGGING:
            autogen.runtime_logging.stop()
        return True, os.path.join(WORKING_DIR, f"{self._problem_id}_generated_code.txt")

    

 