import subprocess
from pathlib import Path

def main():
    results = []

    for p_program in Path("programs").glob("**/*.py"):
        p_program = str(p_program)
        problem = p_program[len("programs/") : -len(".py")]
        p_program_out = "programs/" + problem + ".out"
        p_in = "dataset/" + problem + ".in"
        p_out = "dataset/" + problem + ".out"
        run_str = f"python {p_program} < {p_in} > {p_program_out}"
        print(run_str)
        subprocess.run(run_str, shell=True)

        with open(p_program_out, "r") as f_program_out:
            program_out = f_program_out.readlines()
        if len(program_out) == 0:
            results.append((problem, "FAIL"))
            continue

        with open(p_out, "r") as f_out:
            out = f_out.readlines()
        assert(len(program_out) == len(out))

        good = 0
        for i in range(len(out)):
            if program_out[i] == out[i]:
                good += 1
 
        results.append((problem, f"{good}/{len(out)}"))

    print("| Problem | Score |")
    print("| ------- | ----- |")
    for p, score in results:
        print(f"| {p} | {score} |")

if __name__ == "__main__":
    main()
