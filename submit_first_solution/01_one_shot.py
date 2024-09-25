import asyncio
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional
import logging
from tempfile import TemporaryDirectory

import openai
import weave
import simple_parsing
import instructor
from pydantic import BaseModel, Field

from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, check_correctness, setup_logger, run_program

client = instructor.from_openai(openai.OpenAI())

class Solution(BaseModel):
    solution_explanation: str = Field(..., description="Explanation of the solution to the problem")
    source_code: str = Field(..., description="Valid Python3 sourcecode to solve the problem.")

    def save_code(self, out_file="solution.py"):
        out_file = Path(out_file)
        out_file.write_text(self.source_code)
        return out_file


@weave.op
def call_model(messages, **kwargs):
    response_model = kwargs.pop("response_model", None)
    res = client.chat.completions.create(
        messages=messages,
        response_model=response_model,
        **kwargs
    )
    if response_model is not None:
        return res
    else:
        return res.choices[0].message.content

@weave.op
def generate_code(
    problem: Problem, 
    system_prompt: str, 
    prompt_template: str,
    model: str, 
    use_images: bool = False) -> str:
    logging.info(f"Generating code solution for: {problem.name}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_template.format(
                problem_description=problem.problem_description,
                sample_input=problem.get_sample_input(),
                sample_output=problem.get_sample_output(),
            )}
        ] + ([{"type": "image_url", "image_url": {"url": img}} for img in problem.images] if use_images else [])}
    ]

    # call model one first time to get the code
    logging.info("Generating initial analysis and solution")
    out = call_model(messages=messages, model=model, response_model=None)
    # call model second time to extract the code
    logging.info("  Extracting the code from the previous generation...")
    solution = call_model(
        messages=[{
            "role": "user",
            "content": f"Extract the relevant information from the following document and return it in valid JSON\n\n{out}",
            }], 
        model="gpt-4o", # hard coded for the extraction
        response_model=Solution, 
        max_retries=2
    )

    # in case we have ```python stuff...`
    solution.source_code = maybe_remove_backticks(solution.source_code)
    return solution

system_prompt = "You are an expert problem solver. Your task is creating the code to solve the problem at hand in python."

prompt_template = """
## Problem: 
{problem_description}

## Input: 
{sample_input}

## Output: 
{sample_output}

Create a python program that returns the correct output for the given input. 

## Competition Guidelines:
    a. Do not use any external libraries; stick to Python 3 standard library
    b. Handle input and output using standard input/output (stdin/stdout)
    c. Use helper functions to improve readability of the code.
    c. Use the `input()` function to take input from stdin and print the output to stdout.
    d. Do not add extra print statements otherwise it will fail the test cases.
    e. Make sure your code passes all potential test cases, including edge cases
    f. Follow the input/output format specified in the problem statement and the sample test cases.
    g. We will run the program by calling `python3 program.py` so make sure it outputs the correct results.
"""

class RunAndTestResult(BaseModel):
    correct: bool = Field(..., description="Whether the generated output matches the ground truth output")
    runnable: bool = Field(..., description="Whether the program can run without errors")
    error: Optional[str] = Field(..., description="Error message if the program failed to run")


@weave.op
async def run_and_test(code: str, input_file: Path, output_file: Path, generated_output_file: Path, timeout: int = 30):
    """
    Run the program and test the output against the sample output.
    
    Args:
        code: The path to the code file.
        input_file: The path to the input file.
        output_file: The path to the ground truth output file.
        generated_output_file: The path to the file where the generated output will be saved.
        timeout: The timeout for the code execution.
    """
    try:
        await run_program(code, input_file, generated_output_file, timeout=timeout)
    except Exception as e:
        logging.error(f"Error running program: {e}")
        return RunAndTestResult(correct=False, runnable=False, error=str(e))
    
    correct = check_correctness(output_file.read_text(), generated_output_file.read_text())
    return RunAndTestResult(correct=correct, runnable=True, error=None)

@dataclass
class Args(simple_parsing.Serializable):
    problem_name: str = "cheeseburger_corollary_ch1" # name of the problem to solve
    folder_path: Path = Path("./dataset/2023/practice/") # path to the folder containing the problems
    model: str = "gpt-4o" # openai model to use
    use_images: bool = False # set to True to use images in the prompt
    save_output: bool = True # set to True to save the output to a file
    debug: bool = False # set to True to see the debug logs
    timeout: int = 60 # timeout for the code execution

if __name__=="__main__":
    args = simple_parsing.parse(Args)

    setup_logger(args.debug)

    problem = Problem.from_name(args.problem_name, args.folder_path)

    weave.init("hack-starter")
    
    logging.info("> Solving on sample input...")
    solution = generate_code(
        problem, 
        system_prompt=system_prompt, 
        prompt_template=prompt_template,
        model=args.model,
        use_images=args.use_images)
    
    code_file = solution.save_code(problem.folder_path / (problem.name + "_generated.py"))

    generated_output_file = problem.folder_path / (problem.name + "_generated.out")
    logging.info("> Running and testing the solution on sample input/output...")
    run_and_test_result = asyncio.run(run_and_test(
        code_file, 
        problem.sample_input, 
        problem.sample_output, 
        generated_output_file,
        timeout=args.timeout))

    logging.info(f"> Test sample output: {run_and_test_result}")

    if run_and_test_result.correct:
        logging.info("> Solution is correct!. Solving for full input...")
        asyncio.run(run_program(code_file, problem.input_path, generated_output_file, timeout=args.timeout))
        logging.info(f"Code file: [cyan]{code_file}[/cyan]\nOutput file: [cyan]{generated_output_file}[/cyan]")
    else:
        logging.info("> Solution is incorrect!. Not running on full input.")