from pathlib import Path

import openai
import weave

from utils import Problem

client = openai.OpenAI()


@weave.op
def call_model(messages, **kwargs):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        **kwargs
    ).choices[0].message.content

@weave.op
def generate_solution(problem: Problem, system_propmt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_template.format(
                problem_description=problem.problem_description,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output,
                input_path=problem.input_path,
            )},
            *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in problem.images]
        ]}
    ]
    out = call_model(messages=messages)
    print(out)

    # Let's make a second call to the model to extract the code from the response

    messages.append({"role": "user", "content": [
        {"type": "text", "text": "Extract the code from the response. reply with the code only. Don't add any comments or other text to the code. The code should be a valid python program."}
    ]})


    solution = call_model(messages=messages)
    print(solution)
    return solution

@weave.op
def write_and_run(path: Path, solution: str):
    # save code to solution.py file in the same folder
    with open(path, "w") as f:
        f.write(solution)

    # let's run the solution code against the input file
    exec(open(path).read())


if __name__=="__main__":

    system_prompt = "You are an expert problem solver. Your task is creating the code to solve the problem at hand in python."

    prompt_template = """
    Problem: 
    {problem_description}

    Input: 
    {sample_input}

    Output: 
    {sample_output}

    Create a python program that returns the correct output for the given input. The file should have at the end a __main__ that looks like this:
    ```python
    if __name__ == "__main__":
        with open(f"{input_path}", "r") as f:
            input_data = f.read()
        generated_solution = solve(input_data)
        with open("generated_solution.out", "w") as f:
            f.write(generated_solution)
    ```
    """


    SAMPLE_PATH = Path("assets/cheeseburger_corollary_1")
    problem = Problem.from_folder(SAMPLE_PATH)

    weave.init("hack-starter")

    solution = generate_solution(problem, system_prompt)
    write_and_run(SAMPLE_PATH / "solution.py", solution)



