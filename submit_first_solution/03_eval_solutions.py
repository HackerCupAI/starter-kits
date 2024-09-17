import asyncio
import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path

import weave
import simple_parsing
from pydantic import BaseModel, Field
from rich.logging import RichHandler


def setup_logger(debug=False):
    level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )


class Problem(BaseModel):
    problem_dir: Path = Field(..., description="The path to the problem directory")
    problem_name: str = Field(..., description="The name of the problem")
    problem_description: str = Field(..., description="The description of the problem")
    sample_input: str = Field(
        ..., description="The path to the sample input of the problem"
    )
    sample_output: str = Field(
        ..., description="The path to the sample output of the problem"
    )
    code: str = Field(..., description="The path to the code file")
    input: str = Field(..., description="The path to the input file")
    output: str = Field(..., description="The path to the output file")


def guess_code_file(problem_name: str, problem_dir: Path) -> Path:
    if os.path.exists(problem_dir / f"{problem_name}.cpp"):
        return problem_dir / f"{problem_name}.cpp"
    elif os.path.exists(problem_dir / f"{problem_name}.py"):
        return problem_dir / f"{problem_name}.py"
    else:
        raise ValueError(f"No code file found for problem {problem_name}")


def load_problem(problem_name: str, problem_dir: Path) -> Problem:
    input = problem_dir / f"{problem_name}.in"
    output = problem_dir / f"{problem_name}.out"
    sample_input = problem_dir / f"{problem_name}_sample_input.txt"
    sample_output = problem_dir / f"{problem_name}_sample_output.txt"
    code = guess_code_file(problem_name, problem_dir)
    problem_description = problem_dir / f"{problem_name}.md"
    return Problem(
        problem_dir=problem_dir,
        problem_name=problem_name,
        problem_description=problem_description.read_text(),
        sample_input=str(sample_input),
        sample_output=str(sample_output),
        input=str(input),
        output=str(output),
        code=str(code),
    )


def find_problems(folder: Path) -> list[dict]:
    """
    Find all the problems in the given folder.
    """
    problems = []

    # search for all files ending in .in
    problem_names = [file.stem for file in folder.glob("**/*.in")]
    for problem_name in problem_names:
        try:
            problems.append(load_problem(problem_name, folder))
        except Exception as e:
            logging.error(f"Error loading problem {problem_name}: {e}")
    logging.info(f"Found {len(problems)} problems")
    return problems


async def run_python(
    program: Path, input_file: Path, output_file: Path, timeout: float = 10
):
    """
    Run a Python program with the given input file and output file.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            program,
            stdin=input_file.open("rb"),
            stdout=output_file.open("wb"),
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Program execution timed out after {timeout} seconds")

        if process.returncode != 0:
            raise RuntimeError(f"Program execution failed: {stderr.decode()}")

        logging.info(f"Output saved to {output_file}")
    except Exception as e:
        raise RuntimeError(f"Error running Python program: {str(e)}")


async def run_cpp(
    cpp_file: Path, input_file: Path, output_file: Path, timeout: float = 10,
    cpp_version: int = 11,
):
    """
    Run a C++ program with the given input file and output file.
    """
    # Get the base name of the cpp file (without extension)
    base_name = os.path.splitext(cpp_file.name)[0]

    # Compile the C++ program
    compile_command = f"g++ {cpp_file} -std=c++{cpp_version} -o {base_name}"
    process = await asyncio.create_subprocess_shell(
        compile_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Compilation failed: {stderr.decode()}")

    try:
        # Run the compiled program with input from file
        with open(input_file, "r") as infile:
            process = await asyncio.create_subprocess_exec(
                f"./{base_name}",
                stdin=infile,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError(
                    f"Program execution timed out after {timeout} seconds"
                )

            if process.returncode != 0:
                raise RuntimeError(f"Program execution failed: {stderr.decode()}")

            output_file.write_text(stdout.decode())

        logging.info(f"Output saved to {output_file}")

    finally:
        # Clean up the compiled file
        if os.path.exists(base_name):
            os.remove(base_name)


@weave.op
async def run_program(code: Path, input: Path, output: Path, timeout: float = 10, cpp_version: int = 11):
    try:
        if code.suffix == ".cpp":
            logging.info(f"Running C++ program: {code}")
            await run_cpp(code, input, output, timeout, cpp_version)
        elif code.suffix == ".py":
            logging.info(f"Running Python program: {code}")
            await run_python(code, input, output, timeout)
        else:
            raise ValueError(f"Unsupported file type: {code}")
    except Exception as e:
        raise e
    return


@weave.op
def check_solution(model_output: dict, output: str):
    "A simple check to see if the output is correct"
    # these may be big!
    generated_output = Path(model_output["generated_output"]).read_text()
    output = Path(output).read_text()
    return {
        "solved": generated_output.strip() == output.strip(),
        "runnable": model_output["runnable"],
    }


@weave.op
async def run_and_save_output(code: str, input: str, suffix: str, timeout: float, cpp_version: int = 11) -> dict:
    """
    Run the program and save the output to a file.
    """
    code, input = Path(code), Path(input)
    generated_output = input.parent / (input.stem + suffix)
    try:
        await run_program(code, input, generated_output, timeout=timeout, cpp_version=cpp_version)
    except Exception as e:
        generated_output.write_text(str(e))
        return {
            "generated_output": generated_output,
            "runnable": False,
            "error": str(e),
        }
    return {"generated_output": generated_output, "runnable": True, "error": None}

@dataclass
class Args(simple_parsing.Serializable):
    code: str = "dataset/2023/practice/cheeseburger_corollary_ch1.cpp" # The file to run
    input: str = "dataset/2023/practice/cheeseburger_corollary_ch1.in" # The input to run the program on
    output: str = "dataset/2023/practice/cheeseburger_corollary_ch1.out" # The output to compare against
    eval_name: str = "super_dupper_model" # The name of the evaluation
    weave_project: str = "hackercup-eval-solution" # The name of the weave project
    timeout: float = 30 # The timeout for the program execution (per problem)
    suffix: str = "_generated_output.txt" # The suffix for the generated output file
    verbose: bool = False # Whether to print verbose output
    folder: str = None # Run all problems in this folder
    run_samples: bool = False # Whether to run on the sample input/output pairs
    cpp_version: int = 11 # The C++ version to use for the program execution


if __name__ == "__main__":
    args = simple_parsing.parse(Args)
    setup_logger(args.verbose)

    weave.init(args.weave_project)

    # run one file
    if not args.folder:
        logging.info(f"Running file: {args.code}")
        out = asyncio.run(
            run_and_save_output(args.code, args.input, args.suffix, args.timeout, args.cpp_version)
        )

        passed = check_solution(out, args.output)
        logging.info(f"Program passed: {passed}")
    
    else:
        logging.info(f"Running folder: {args.folder}")
        logging.info("=" * 60)
        problems = find_problems(Path(args.folder))

        class Runner(weave.Model):
            timeout: float = 10
            suffix: str = "_generated_output.txt"
            cpp_version: int = 11

            @weave.op
            async def predict(self, code: str, input: str):
                return await run_and_save_output(code, input, self.suffix, self.timeout, self.cpp_version)

        dataset = [
            {
                "input": problem.sample_input if args.run_samples else problem.input,
                "output": problem.sample_output if args.run_samples else problem.output,
                "code": problem.code,
                "problem_name": problem.problem_name,
            }
            for problem in problems
        ]

        model = Runner(timeout=args.timeout, suffix=args.suffix, cpp_version=args.cpp_version)
        evaluation = weave.Evaluation(name=args.eval_name, dataset=dataset, scorers=[check_solution])
        asyncio.run(evaluation.evaluate(model))
