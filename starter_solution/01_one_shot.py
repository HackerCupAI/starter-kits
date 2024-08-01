from dataclasses import dataclass
from pathlib import Path

import openai
import weave
import simple_parsing

from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution

client = openai.OpenAI()


@weave.op
def call_model(messages, **kwargs):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        **kwargs
    ).choices[0].message.content

@weave.op
def generate_code(problem: Problem, system_prompt: str, prompt_template: str) -> str:
    print(f"Generating code solution for: {problem.name}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_template.format(
                problem_description=problem.problem_description,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output,
            )},
            *[{"type": "image_url", 
               "image_url": {"url": img}} for img in problem.images]
        ]}
    ]

    # call model one first time to get the code
    out = call_model(messages=messages)
    print("Generating initial analysis and solution")

    # Let's make a second call to the model to extract the code from the response
    messages.append({"role": "user", "content": [
        {"type": "text", 
         "text": ("Extract the code from the response. reply with the code only. "
                  "Don't add any comments or other text to the code. "
                  "The code should be a valid python program.")}
    ]})

    # call model second time to extract the code
    solution = call_model(messages=messages)
    print("Extracting the solution from the previous generation...")

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
input: The same Input provided above
output: The same Output provided above

```python
def solve(input: str) -> str: 
```
"""

@dataclass
class Args(simple_parsing.Serializable):
    problem_name: str = "cheeseburger_corollary_ch1"
    folder_path: Path = Path("./dataset/2023/practice/")
    log: bool = False

if __name__=="__main__":
    args = simple_parsing.parse(Args)

    problem = Problem.from_name(args.problem_name, args.folder_path)

    if args.log: weave.init("hack-starter")

    code = generate_code(problem, system_prompt, prompt_template)

    sample_output = problem.exec(code, input=problem.sample_input)
    sample_matches = check_solution(problem.sample_output, sample_output)
    print("Sample Matches:")
    print(sample_matches)

    # now against the real input
    output = problem.exec(code, input=problem.input)
    matches = check_solution(problem.output, output)
    print("Final Matches:")
    print(matches)


