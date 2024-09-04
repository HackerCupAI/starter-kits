import asyncio
import logging
from typing import List

import weave

from retriever import Retriever, rerank_docs
from utils import (
    FAST_LLM,
    STRONG_LLM,
    Analysis,
    Problem,
    Reflection,
    Solution,
    async_client,
    check_correctness,
    format_example,
    format_examples,
    format_response,
)

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


SOLVER_INSTRUCTIONS = """You are a world-class competitive programmer tasked with solving a programming problem. 
You will be provided with a problem statement, and you need to create a Python3 solution for it. 
Your task it to develop a winning solution to the problem in Python3 programming language.
You will do this in a step-by-step manner.

Step 1: Extract the core question and the problem-solving information from the problem statement.
Step 2: Describe the algorithm used to solve the problem.
Step 3: Write a short tutorial on the algorithm and how it works.
Step 4: Generate a step by step plan to solve the problem.
Step 5: Generate the pseudocode to solve the problem.
Step 6: Write the final solution in Python3 programming language to solve the problem.

Competition Guidelines:
    a. Do not use any external libraries; stick to Python 3 standard library
    b. Handle input and output using standard input/output (stdin/stdout)
    c. Use helper functions to improve readability of the code.
    c. Use the `input()` function to take input from stdin and print the output to stdout.
    d. Do not add extra print statements otherwise it will fail the test cases.
    e. Make sure your code passes all potential test cases, including edge cases
    f. Follow the input/output format specified in the problem statement and the sample test cases.


**Formatting Instructions: Your response must follow the following xml format** -

<root>
<core_question>
[Extract core question, only the most comprehensive and detailed one!]
</core_question>
<problem_solving_info>
[Extract problem-solving information related to the core question, only the most comprehensive and detailed one!]
</problem_solving_info>
<algorithm>
[Algorithm to solve the problem. Describe the algorithm used to solve the problem such that a novice programmer without any prior knowledge of the solution can implement it. Do not generate code.]
</algorithm>
<tutorial>
[Write a useful tutorial about the above mentioned algorithm(s). Provide a high level generic tutorial for solving these types of problems. Do not generate code.]
</tutorial>
<plan>
[Generate a step by step plan to solve the problem.]
</plan>
<pseudocode>
[Generate a pseudocode to solve the problem.]
</pseudocode>
<source_code>
[Write the final solution in Python3 programming language to solve the problem.]
</source_code>
</root>

---
"""


@weave.op
async def draft_solution(
    problem: Problem, model: str = FAST_LLM, temperature: float = 0.0
) -> Solution:
    user_prompt = f"""{problem.as_xml}
---
Let's think step by step to solve the problem:
"""

    response = await async_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SOLVER_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        response_model=None,
        temperature=temperature,
    )
    formatted_response = await format_response(
        response.choices[0].message.content, Solution
    )
    return formatted_response


ANALYSIS_INSTRUCTIONS = """You are an expert programming analyst with a deep understanding of competitive programming.
You are provided with a problem statement and a solution to a problem.
Your task is to develop a step by step plan and pseudocode to solve the problem.
You will do this in a step by step manner.
First, extract the core question and the problem-solving information from the problem statement.
Then, describe the algorithm used to solve the problem.
Then, write a short tutorial on the algorithm and how it works.
Next, generate a step by step plan to solve the problem.
Finally, generate the pseudocode to solve the problem.

**Formatting Instructions: Your response must follow the following xml format** -

<root>
<core_question>
[Extract core question, only the most comprehensive and detailed one!]
</core_question>
<problem_solving_info>
[Extract problem-solving information related to the core question, only the most comprehensive and detailed one!]
</problem_solving_info>
<algorithm>
[Algorithm to solve the problem. Describe the algorithm used to solve the problem such that a novice programmer without any prior knowledge of the solution can implement it. Do not generate code.]
</algorithm>
<tutorial>
[Write a useful tutorial about the above mentioned algorithm(s). Provide a high level generic tutorial for solving these types of problems. Do not generate code.]
</tutorial>
<plan>
[Generate a step by step plan to solve the problem.]
</plan>
<pseudocode>
[Generate a pseudocode to solve the problem.]
</pseudocode>
</root>
"""


@weave.op
async def describe_example(example: dict) -> Analysis:
    user_prompt = f"""{format_example(example)}

Let's think step by step to analyze the problem and plan a solution to the problem:
"""
    response = await async_client.chat.completions.create(
        model=FAST_LLM,
        messages=[
            {"role": "system", "content": ANALYSIS_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        response_model=None,
    )

    formatted_response = await format_response(
        response.choices[0].message.content, Analysis
    )
    return formatted_response


@weave.op
async def describe_examples(docs: List[dict]) -> List[Analysis]:
    tasks = []
    for doc in docs:
        tasks.append(describe_example(doc))
    descriptions = await asyncio.gather(*tasks)
    return descriptions


@weave.op
async def generate_solution(
    problem: Problem, examples: str, model: str = STRONG_LLM, temperature: float = 0.0
) -> Solution:
    instructions_prompt = f"""{SOLVER_INSTRUCTIONS}

You have previously solved the following problems in this competition:
<examples>
{examples}
</examples>
"""

    messages = [
        {"role": "system", "content": instructions_prompt},
        {
            "role": "user",
            "content": f"""{problem.as_xml}
---
Let's think step by step to solve the problem:

""",
        },
    ]
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=None,
        temperature=temperature,
    )

    formatted_response = await format_response(
        response.choices[0].message.content, Solution
    )
    return formatted_response


REFLECTION_INSTRUCTIONS = """You are a world-class competitive programmer with a keen eye for detail and problem solving. 
Your expertise is in algorithms and data structures. 
You have incorrectly answered the following programming problem. 
Your task is to reflect on the problem, your solution, and the correct answer.
You will then use this information help you answer the same question in the future. 
First, explain why you answered the question incorrectly.
Second, list the keywords that describe the type of your errors from most general to most specific.
Third, solve the problem again, step-by-step, based on your knowledge of the correct answer.
Fourth, create a list of detailed instructions to help you correctly solve this problem in the future.
Finally, create a list of general advice to help you solve similar types of problems in the future.
Be concise in your response; however, capture all of the essential information.

{problem}
<incorrect_solution>
{incorrect_solution}
</incorrect_solution>
<test_report>
{test_report}
</test_report>

**Format Instructions: Your response must follow the following xml format** -

<root>
<reflection>
[Reflect on the problem, your solution, and the correct answer.]
</reflection>
<keywords>
[List the keywords that describe the type of your errors from most general to most specific.]
</keywords>
<step_by_step_solution>
[Solve the problem again, step-by-step, based on your knowledge of the correct answer.]
</step_by_step_solution>
<instructions>
[Create a list of detailed instructions to help you correctly solve this problem in the future.]
</instructions>
<general_advice>
[Create a list of general advice to help you solve similar types of problems in the future.]
</general_advice>
</root>
---
Let's think step by step to reflect on the problem:
"""


@weave.op
async def reflection(
    problem: Problem,
    incorrect_solution: Solution,
    test_report: str,
    model: str = STRONG_LLM,
    temperature: float = 0.0,
) -> Reflection:
    system_prompt = REFLECTION_INSTRUCTIONS.format(
        problem=problem.as_xml,
        incorrect_solution=incorrect_solution.as_xml,
        test_report=test_report,
    )
    messages = [{"role": "system", "content": system_prompt}]
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=None,
        temperature=temperature,
    )
    formatted_response = await format_response(
        response.choices[0].message.content, Reflection
    )
    return formatted_response


@weave.op
async def improve_solution(
    problem: Problem,
    incorrect_solution: Solution,
    test_report: str,
    reflections: Reflection,
    model: str = STRONG_LLM,
    temperature: float = 0.0,
) -> Solution:
    messages = [
        {"role": "system", "content": SOLVER_INSTRUCTIONS},
        {"role": "user", "content": problem.as_xml},
        {"role": "assistant", "content": incorrect_solution.as_xml},
        {"role": "user", "content": f"<test_report>\n{test_report}\n</test_report>"},
        {
            "role": "user",
            "content": "Your previous answer to the question is incorrect. Please reflect on the problem, your solution, and the correct answer.",
        },
        {"role": "assistant", "content": reflections.as_xml},
        {
            "role": "user",
            "content": """Use your self-reflection (above) to help you answer the question correctly.

---
Let's think step by step to solve the problem correctly:
""",
        },
    ]
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=None,
        temperature=temperature,
    )
    formatted_response = await format_response(
        response.choices[0].message.content, Solution
    )
    return formatted_response


@weave.op
async def zero_shot_solver(
    problem: Problem, model: str = FAST_LLM, temperature: float = 0.0, timeout: int = 10
) -> dict:
    logger.info("Drafting intial zero-shot solution")
    solution = await draft_solution(
        problem=problem,
        model=model,
        temperature=temperature,
    )
    test_report = check_correctness(
        solution.source_code, problem.sample_input, problem.sample_output, timeout
    )
    logger.info(f"Draft solution result: {repr(test_report)}")
    return {"solution": solution, "test_report": test_report, "stage": "zero-shot"}


@weave.op
async def rag_solver(
    retriever: Retriever,
    problem: Problem,
    model: str = FAST_LLM,
    temperature: float = 0.0,
    timeout: int = 10,
) -> dict:
    zero_shot_result = await zero_shot_solver(
        problem=problem,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )
    solution = zero_shot_result["solution"]
    test_report = zero_shot_result["test_report"]
    if test_report == "passed":
        return zero_shot_result
    logger.info("Iterating on a RAG solution")

    @weave.op
    async def create_examplars(
        problem: Problem, solution: Solution, top_k: int = 50, top_n: int = 5
    ):
        logger.info(f"Generating examplars:")
        retrieve_docs = retriever.retrieve(solution.source_code, top_k)
        reranked_docs = await rerank_docs(problem, solution, retrieve_docs, top_n)
        analyses = await describe_examples(reranked_docs)
        examplars = format_examples(reranked_docs, analyses)
        return examplars

    @weave.op
    async def rag_solution(
        problem: Problem,
        draft_solution: Solution,
        model: str = STRONG_LLM,
        temperature: float = 0.0,
        timeout: int = timeout,
    ) -> dict:
        logger.info(f"Generating RAG solution:")
        examplars = await create_examplars(problem, draft_solution)
        rag_solution = await generate_solution(
            problem=problem,
            examples=examplars,
            model=model,
            temperature=temperature,
        )
        test_report = check_correctness(
            rag_solution.source_code,
            problem.sample_input,
            problem.sample_output,
            timeout,
        )
        logger.info(f"RAG Solution Result: {repr(test_report)}")
        return {"solution": rag_solution, "test_report": test_report}

    rag_result = await rag_solution(problem, solution, model, temperature, timeout)
    solution = rag_result["solution"]
    test_report = rag_result["test_report"]
    return {"solution": solution, "stage": "rag", "test_report": test_report}


@weave.op
async def rework_solution(
    problem: Problem,
    incorrect_solution: Solution,
    test_report: str,
    model: str = STRONG_LLM,
    temperature: float = 0.0,
    timeout: int = 10,
) -> dict:
    logger.info(f"Reflecting and improving solution")
    reflections = await reflection(
        problem=problem,
        incorrect_solution=incorrect_solution,
        test_report=test_report,
        model=model,
        temperature=temperature,
    )
    improved_solution = await improve_solution(
        problem=problem,
        incorrect_solution=incorrect_solution,
        test_report=test_report,
        reflections=reflections,
        model=model,
        temperature=temperature,
    )
    test_report = check_correctness(
        improved_solution.source_code,
        problem.sample_input,
        problem.sample_output,
        timeout,
    )
    logger.info(f"Reworked solution result: {repr(test_report)}")
    return {"solution": improved_solution, "test_report": test_report}


@weave.op
async def rag_solver_with_reflection(
    retriever: Retriever,
    problem: Problem,
    model: str = FAST_LLM,
    temperature: float = 0.0,
    max_iterations: int = 2,
    timeout: int = 10,
):
    num_iterations = 0
    test_report = "failed"
    solution = None
    while not test_report == "passed" and num_iterations < max_iterations:
        rag_result = await rag_solver(
            retriever=retriever,
            problem=problem,
            timeout=timeout,
            model=model,
            temperature=temperature,
        )
        solution = rag_result["solution"]
        test_report = rag_result["test_report"]
        if test_report == "passed":
            return rag_result
        rework_result = await rework_solution(
            problem=problem,
            incorrect_solution=solution,
            test_report=test_report,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
        solution = rework_result["solution"]
        test_report = rework_result["test_report"]
        if test_report == "passed":
            return {
                "solution": solution,
                "stage": "reflection",
                "test_report": test_report,
            }
        num_iterations += 1
    logger.info("Failed to generate a solution")
    return {"solution": solution, "stage": "failed", "test_report": test_report}
