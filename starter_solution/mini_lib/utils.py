import logging
from typing import Optional
from rich.logging import RichHandler

def setup_logger(debug = False, silence_openai = True):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    # silence openai logger
    if silence_openai:
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

def maybe_remove_backticks(solution: str) -> str:
    "Remove backticks from the solution"
    if solution.startswith("```python"):
        solution = solution[len("```python") :]
    if solution.endswith("```"):
        solution = solution[: -len("```")]
    return solution

def check_solution(expected: str, actual: str) -> dict:
    "Check the solution against the expected output"
    matches = 0
    expected_lines = expected.split("\n")
    logging.debug(f"Expected lines: {expected_lines}")
    actual_lines = actual.split("\n")
    logging.debug(f"Actual lines: {actual_lines}")
    offending_cases = []
    for expected_line, actual_line in zip(expected_lines, actual_lines):
        expected_line = expected_line.strip()
        actual_line = actual_line.strip()
        
        if expected_line == actual_line:
            matches += 1  # +1 for the whole line match
        else:
            offending_cases.append((expected_line, actual_line))
    return {"matches": matches, "total": len(expected_lines), "offending_cases": offending_cases}

def run(code: Optional[str] = None, input: Optional[str] = None):
    logging.info("Running solution synchronously...")
    vars = {}
    try:
        exec(code, vars)
    except Exception as e:
        logging.error(f"The generated code is not valid: {code}")
        raise e
    try:
        fn = vars.get("solve", lambda x: x)
        return fn(input)
    except Exception as e:
        logging.error(f"Error executing code: {e}")
        raise e

if __name__ == "__main__":
    # Test check_solution
    expected = "Case #1: YES\nCase #2: NO\nCase #3: YES"
    actual = "Case #1: YES\nCase #2: Yes\nCase #3: YES"
    result = check_solution(expected, actual)
    assert result["matches"] == 2, "Expected 2 matches"
    assert result["total"] == 3, "Expected 3 total lines"
    assert len(result["offending_cases"]) == 1, "Expected 1 offending case"
    assert result["offending_cases"][0] == ("Case #2: NO", "Case #2: Yes"), "Unexpected offending case"

    # Test maybe_remove_backticks
    assert maybe_remove_backticks("print('hello')\n```") == "print('hello')\n", "Failed to remove trailing backticks"
    assert maybe_remove_backticks("```python\nprint('hello')") == "\nprint('hello')", "Failed to remove leading backticks"
    assert maybe_remove_backticks("```python\nprint('hello')\n```") == "\nprint('hello')\n", "Failed to remove both leading and trailing backticks"

    # test exec
    code = "def solve(x):\n    return x + 1"
    input = "2"
    result = run(code, input)
    assert result == 3, "Expected 3"
    print("All tests passed!")

