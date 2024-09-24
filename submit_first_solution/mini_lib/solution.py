from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional


class Solution(BaseModel):
    problem_name: str = Field(..., description="The name of the problem")
    problem_folder: str | Path = Field(..., description="The folder of the problem")
    solution_explanation: str = Field(..., description="Explanation of the solution to the problem")
    source_code: str = Field(..., description="Valid Python3 sourcecode to solve the problem.")
    code_path: Optional[str] = Field(None, description="The path to the code file")
    outfile: Optional[str] = Field(None, description="The path of the output file")

    def save_code(self, code: str, code_path: Optional[str] = None, outfile: Optional[str] = None):
        code_name = f"{self.problem_name}_generated.py"
        code_path = Path(self.problem_folder) / code_name if code_path is None else code_path
        outfile = f"./{self.problem_name}_generated.out" if outfile is None else outfile
        code_path.write_text(code)
        return Path(code_path)

    def save_output(self, output: str, outfile: Optional[str] = None, suffix: str = "_generated.out"):
        outfile_name = f"{self.problem_name}{suffix}"
        outfile = Path(self.problem_folder) / outfile_name if outfile is None else outfile
        outfile.write_text(output)
        return Path(outfile)
    
    def __str__(self):
        return (
            f"Problem Name: {self.problem_name}\n"
            f"Problem Folder: {self.problem_folder}\n"
            f"Solution Explanation: {self.solution_explanation}\n"
            f"Source Code: {self.source_code}\n"
            f"Code Path: {self.code_path}\n"
            f"Outfile: {self.outfile}\n"
        )


if __name__ == "__main__":
    problem_name ="cheeseburger_corollary_ch1"
    problem_folder = Path("../dataset/2023/practice/")
    solution = Solution(
        problem_name=problem_name,
        problem_folder=problem_folder,
        solution_explanation="",
        source_code="",
        code_path=None,
        outfile=None
    )
    print(solution)