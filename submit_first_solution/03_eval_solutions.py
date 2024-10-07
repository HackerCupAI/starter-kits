import asyncio
import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path

import weave
import simple_parsing

from mini_lib.problem import find_problems
from mini_lib.utils import setup_logger, run_program, check_correctness, prepare_problems


@weave.op
async def run_and_save_output(code: str, input: str, suffix: str, timeout: float, cpp_version: int = 11) -> dict:
    """
    Run the program and save the output to a file using `run_program` from `mini_lib`.
    """
    code, input = Path(code), Path(input)
    generated_output = input.parent / (input.stem + suffix)
    try:
        await run_program(code, input, generated_output, timeout=timeout, cpp_version=cpp_version)
        runnable = True
        error = None
    except Exception as e:
        generated_output.write_text(str(e))
        runnable = False
        error = str(e)
    return {
        "generated_output": generated_output,
        "runnable": runnable,
        "error": error,
    }


@weave.op
def check_solution(model_output: dict, output: Path):
    "Check if the generated output matches the expected output."
    generated_output = Path(model_output["generated_output"]).read_text()
    expected_output = Path(output).read_text()
    solved = check_correctness(expected_output, generated_output)
    return {
        "solved": solved,
        "runnable": model_output["runnable"],
    }


@dataclass
class Args(simple_parsing.Serializable):
    code: str = "dataset/2023/practice/cheeseburger_corollary_ch1.cpp" # The file to run
    input: str = "dataset/2023/practice/cheeseburger_corollary_ch1.in" # The input to run the program on
    output: str = "dataset/2023/practice/cheeseburger_corollary_ch1.out" # The output to compare against
    eval_name: str = "super_dupper_model" # The name of the evaluation
    weave_project: str = "hackercup-eval-solution" # The name of the weave project
    timeout: float = 30 # The timeout for the program execution (per problem)
    suffix: str = "_generated.txt" # The suffix for the generated output file
    verbose: bool = False # Whether to print verbose output
    folder: Path = None # Run all problems in this folder
    run_samples: bool = False # Whether to run on the sample input/output pairs
    cpp_version: int = 11 # The C++ version to use for the program execution


if __name__ == "__main__":
    args = simple_parsing.parse(Args)
    setup_logger(args.verbose)

    weave.init(args.weave_project)

    # Run one file or all problems in a folder
    if not args.folder:
        logging.info(f"Running file: {args.code}")
        out = asyncio.run(
            run_and_save_output(args.code, args.input, args.suffix, args.timeout, args.cpp_version)
        )
        result = check_solution(out, args.output)
        logging.info(f"Program result: {result}")
    else:
        logging.info(f"Preparing problems...")
        prepare_problems(Path(args.folder))
        
        logging.info(f"Running folder: {args.folder}")
        logging.info("=" * 60)
        problems = find_problems(Path(args.folder))

        class Runner(weave.Model):
            timeout: float = args.timeout
            suffix: str = args.suffix
            cpp_version: int = args.cpp_version

            @weave.op
            async def predict(self, code: str, input: str):
                print(f"Running problem: {input}, code: {code}")
                return await run_and_save_output(code, input, self.suffix, self.timeout, self.cpp_version)

        dataset = [
            {
                "input": str(problem.sample_input if args.run_samples else problem.input_path),
                "output": str(problem.sample_output if args.run_samples else problem.output_path),
                "code": str(problem.load_code()),
                "problem_name": problem.name,
            }
            for problem in problems
        ]

        model = Runner()
        evaluation = weave.Evaluation(name=args.eval_name, dataset=dataset, scorers=[check_solution])
        asyncio.run(evaluation.evaluate(model))