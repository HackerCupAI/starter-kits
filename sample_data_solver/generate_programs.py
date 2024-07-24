from pathlib import Path

from transformers import AutoTokenizer
import transformers
import torch

MAX_NEW_TOKENS = 500
MAX_TIME = 120

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
pipeline = transformers.pipeline(
    "text-generation",
    model="codellama/CodeLlama-7b-Python-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)


def generate_func(ins, outs):
    code = "def f(a):\n"
    code += '    """Returns solution\n'
    for i, o in zip(ins, outs):
        code += f"    >>> f({i})\n"
        code += f"    {o}\n"
    code += '    """\n'

    seq = pipeline(
        code,
        do_sample=True,
        temperature=0.1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=MAX_NEW_TOKENS,
        tokenizer=tokenizer,
        stop_strings=["def "],  # stop if second func started
        max_time=MAX_TIME,  # seconds
    )[0]
    full_result = seq["generated_text"]

    # Cut out just the first function
    result = ""
    for i, line in enumerate(full_result.splitlines()):
        if i > 0 and len(line) > 0 and line[0] != " ":
            break
        if line.strip():
            result += f"{line}\n"
    return result


template = """
T = int(input())
for case_num in range(1, T + 1):
    a = input().split()
    for i in range(len(a)):
        try:
            a[i] = float(a[i])
        except ValueError:
            pass
        try:
            a[i] = int(a[i])
        except ValueError:
            pass
    if len(a) == 1:
        a = a[0]
    else:
        if isinstance(a[0], int) and isinstance(a[1], str):
            if (len(a) - 1) == a[0]:
                a = a[1:]
    print(f"Case #{case_num}: {f(a)}")
"""


def process_line(line):
    a = line.split()
    for i in range(len(a)):
        try:
            a[i] = float(a[i])
        except ValueError:
            pass
        try:
            a[i] = int(a[i])
        except ValueError:
            pass
    if len(a) == 1:
        a = a[0]
    else:
        # Simplify when first item is number of strings in list
        if isinstance(a[0], int) and isinstance(a[1], str):
            if (len(a) - 1) == a[0]:
                a = a[1:]
    return a


def get_sample_ins_outs():
    results = []

    # Find problems where each test case
    # is one input line and one output line.
    suitable_problems = []
    for p_in in Path("dataset").glob("**/*_sample_input.txt"):
        with open(p_in, "r") as f:
            num_cases = int(f.readline())
            num_lines = len(f.readlines())

        p_out = str(p_in).replace("input.txt", "output.txt")
        try:
            with open(p_out, "r") as f:
                num_lines_out = len(f.readlines())
            if num_cases == num_lines == num_lines_out:
                suitable_problems.append(p_in)
        except FileNotFoundError:
            pass

    for p_in in suitable_problems:
        max_line_len = 100

        ins = []
        outs = []
        too_large = False
        with open(p_in, "r") as f:
            num_cases = int(f.readline())
            for case_num in range(num_cases):
                line = f.readline()
                if len(line) > max_line_len:
                    too_large = True
                ins.append(process_line(line))

        p_out = str(p_in).replace("input.txt", "output.txt")
        with open(p_out, "r") as f:
            for case_num in range(num_cases):
                line = f.readline()[len("Case #1: ") :]  # Remove Case num prefix
                outs.append(process_line(line))

        if not too_large:
            results.append((p_in, ins, outs))

    return results


def main():
    data = get_sample_ins_outs()
    for p_in, ins, outs in data:
        f = generate_func(ins, outs)
        p_program = "programs/" + str(p_in)[len("dataset/") : -len("_sample_input.txt")] + ".py"
        p_program = Path(p_program)
        p_program.parent.mkdir(parents=True, exist_ok=True)
        p_program.write_text(f + template)


if __name__ == "__main__":
    main()
