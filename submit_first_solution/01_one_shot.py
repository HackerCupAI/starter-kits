from dataclasses import dataclass
from pathlib import Path
import logging

import openai
import weave
import simple_parsing

from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution, setup_logger, run

client = openai.OpenAI()


@weave.op
def call_model(messages, **kwargs):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        **kwargs
    ).choices[0].message.content

@weave.op
def generate_code(
    problem: Problem, 
    system_prompt: str, 
    prompt_template: str, 
    extract_prompt: str,
    use_images: bool = False) -> str:
    logging.info(f"Generating code solution for: {problem.name}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_template.format(
                problem_description=problem.problem_description,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output,
            )}
        ] + ([{"type": "image_url", "image_url": {"url": img}} for img in problem.images] if use_images else [])}
    ]

    # call model one first time to get the code
    out = call_model(messages=messages)
    logging.info("Generating initial analysis and solution")

    # Let's make a second call to the model to extract the code from the response
    messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": [
        {"type": "text", 
         "text": extract_prompt}
    ]})

    # call model second time to extract the code
    solution = call_model(messages=messages)
    logging.info("Extracting the solution from the previous generation...")

    # in case we have ```python stuff...`
    solution = maybe_remove_backticks(solution)
    return solution

system_prompt = "You are an expert problem solver. Your task is creating the code to solve the problem at hand in python."

prompt_template = """
Problem: 
{problem_description}

Input: 
{sample_input}

Output: 
{sample_output}

Create a python program that returns the correct output for the given input. 
The file should have a single `solve` method that has the following signature:
input: [str]: The same Input provided above
output [str]: The same Output provided above

```python
from tqdm import tqdm
def solve(input: str) -> str: 
```
"""

extract_prompt = """
Extract the code from the response. reply with the code only. Omit any additional example or explanation.
- If the solution involves a for loop, please use `for sample in tqdm(range(samples))` to show progress.
- The code should be a valid python program.
- Get the `solve` function with the corresponding imports"""

@weave.op
def solve_problem(problem: Problem, use_images=False, timeout=60) -> dict:
    code = generate_code(
        problem, 
        system_prompt=system_prompt, 
        prompt_template=prompt_template, 
        extract_prompt=extract_prompt, 
        use_images=use_images)
    input, output = problem.sample_input, problem.sample_output
    generated_output = run(code, input=input, timeout=timeout) 
    
    return {"code": code, "generated_output": generated_output, "expected_output": output}

@dataclass
class Args(simple_parsing.Serializable):
    problem_name: str = "cheeseburger_corollary_ch1" # name of the problem to solve
    folder_path: Path = Path("./dataset/2023/practice/") # path to the folder containing the problems
    weave_log: bool = False # set to True to log to weave
    use_images: bool = False # set to True to use images in the prompt
    save_output: bool = True # set to True to save the output to a file
    debug: bool = False # set to True to see the debug logs
    timeout: int = 60 # timeout for the code execution

if __name__=="__main__":
    args = simple_parsing.parse(Args)

    setup_logger(args.debug)

    problem = Problem.from_name(args.problem_name, args.folder_path)

    if args.weave_log: 
        weave.init("hack-starter")
    
    logging.info("> Solving on sample input...")
    problem_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout)
    matches = check_solution(problem_solution["expected_output"], problem_solution["generated_output"])
    logging.info("Sample Matches:")
    logging.info(matches)

    logging.info("> Solving on full input...")
    expected_output = problem.get_output()
    generated_output = run(problem_solution["code"], input=problem.get_input(), timeout=args.timeout) 
    matches = check_solution(expected_output, generated_output)
    logging.info("Final Matches:")
    logging.info(matches)

    if args.save_output:
        logging.info("> Saving output to files")
        problem.save_output(problem_solution["generated_output"])
        problem.save_code(problem_solution["code"])

