import logging
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

if __name__ == "__main__":
    matches = check_solution("Case #1: YES\nCase #2: NO\nCase #3: YES", "Case #1: YES\nCase #2: Yes\nCase #3: YES")
    print(matches)

