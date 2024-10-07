import asyncio
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional
import logging
from tempfile import TemporaryDirectory

import openai
import weave
import simple_parsing
import instructor
from pydantic import BaseModel, Field

from mini_lib.problem import Problem, find_problems
from mini_lib.solution import Solution, ExtractedSolution
from mini_lib.utils import maybe_remove_backticks, check_correctness, setup_logger, run_program

client = instructor.from_openai(openai.OpenAI())


@weave.op
def call_model(messages, **kwargs):
    response_model = kwargs.pop("response_model", str)
    res = client.chat.completions.create(
        messages=messages,
        response_model=response_model,
        **kwargs
    )
    return res


@weave.op
def generate_code(
    problem: Problem, 
    system_prompt: str, 
    prompt_template: str,
    model: str, 
    use_images: bool = False) -> str:
    logging.info(f"Generating code solution for: {problem.name}")



    if "o1" in model:
        messages = [
            {"role": "user", "content": system_prompt + "\n\n" + prompt_template.format(
                problem_description=problem.problem_description,
                sample_input=problem.get_sample_input(),
                sample_output=problem.get_sample_output(),
            )}
        ]
    else:
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
        response_model=ExtractedSolution, 
        max_retries=2
    )
    return Solution(
        source_code=maybe_remove_backticks(solution.source_code),
        solution_explanation=solution.solution_explanation,
        problem_name=problem.name, 
        problem_folder=problem.folder_path,
    )

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
    folder_path: Path = Path("./dataset/2023/practice") # path to the folder containing the problems
    max_num_problems: int = 5 # maximum number of problems to evaluate
    model: str = "gpt-4o" # openai model to use
    use_images: bool = False # set to True to use images in the prompt
    debug: bool = False # set to True to see the debug logs
    timeout: int = 60 # timeout for the code execution

if __name__=="__main__":
    args = simple_parsing.parse(Args)

    setup_logger(args.debug)

    logging.info(f"Parsed args: {args}")
    t0 = time.perf_counter()

    problems = find_problems(args.folder_path)[:args.max_num_problems] # dataset
    logging.info(f"Found {len(problems)} problems")

    weave.init("hack-starter")

    @weave.op
    async def solve_problem(problem: Problem) -> dict:
        """
        Solve the problem and return the result of the run and test.
        """
        solution = generate_code(
            problem, 
            system_prompt=system_prompt, 
            prompt_template=prompt_template,
            model=args.model,
            use_images=args.use_images)
        
        code_file = solution.save_code()
        generated_output_file = problem.folder_path / (problem.name + "_generated.out")
        run_and_test_result = await run_and_test(
            code_file, 
            problem.sample_input, 
            problem.sample_output, 
            generated_output_file,
            timeout=args.timeout)
        return run_and_test_result


    def score(model_output: str):
        return model_output


    dataset = [{"problem": problem} for problem in problems]
    evaluation = weave.Evaluation(dataset=dataset, scorers=[score])
    asyncio.run(evaluation.evaluate(solve_problem))