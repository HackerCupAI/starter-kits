import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import openai
import weave
import simple_parsing
from tqdm.asyncio import tqdm

from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, setup_logger, check_solution

client = openai.AsyncOpenAI()


@weave.op
async def call_model(messages, **kwargs):
    out = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        **kwargs
    )
    return out.choices[0].message.content

@weave.op
async def generate_code(
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
    out = await call_model(messages=messages)
    logging.info("Generating initial analysis and solution")

    # Let's make a second call to the model to extract the code from the response
    messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": [
        {"type": "text", 
         "text": extract_prompt}
    ]})

    # call model second time to extract the code
    solution = await call_model(messages=messages)
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

@dataclass
class Args(simple_parsing.Serializable):
    folder_path: Path = Path("./dataset/2023/practice") # path to the folder containing the problems
    log: bool = False # set to True to log to weave
    max_num_problems: int = 10 # maximum number of problems to evaluate
    on_sample: bool = False # run evaluation on sample inputs/outputs
    use_images: bool = False # set to True to use images in the prompt
    debug: bool = False # set to True to see the debug logs

if __name__=="__main__":
    args = simple_parsing.parse(Args)
    setup_logger(args.debug)
    logging.info(f"Parsed args: {args}")

    problems = Problem.find_all(args.folder_path)  # dataset

    if args.log: weave.init("hack-starter")

    @weave.op
    async def generate_sol(problem: Problem):
        code = await generate_code(
            problem, 
            system_prompt=system_prompt, 
            prompt_template=prompt_template, 
            extract_prompt=extract_prompt, 
            use_images=args.use_images)
        logging.info(f"Generated code: {code}")
        output = problem.exec(code, input=problem.sample_input if args.on_sample else problem.input)
        return output

    def match(problem: Problem, model_output: str):
        matches = check_solution(problem.sample_output if args.on_sample else problem.output, model_output)
        return matches


    if args.log:
        dataset = [{"problem": problem} for problem in problems]
        evaluation = weave.Evaluation(dataset=dataset, scorers=[match])
        asyncio.run(evaluation.evaluate(generate_sol))
    else:
        async def task(problem):
            model_output = await generate_sol(problem)
            matches = match(problem, model_output)
            logging.info(f"Problem {problem.name} results: {matches}")
            return matches

        async def evaluate():
            tasks = [task(problem) for problem in problems]
            eval_results = await tqdm.gather(*tasks, desc="Solving problems...")
            return eval_results

        eval_results = asyncio.run(evaluate())

        # let's format the results in a pandas dataframe
        import pandas as pd
        df = pd.DataFrame(eval_results)
        logging.info(f"Evaluation results: {df}")
