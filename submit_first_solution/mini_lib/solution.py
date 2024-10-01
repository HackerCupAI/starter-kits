from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pathlib import Path
from typing import Optional, Any


class ExtractedSolution(BaseModel):
    solution_explanation: str = Field(..., description="Explanation of the solution to the problem")
    source_code: str = Field(..., description="Valid Python3 sourcecode to solve the problem.")
    
class Solution(ExtractedSolution):
    problem_name: str = Field(..., description="The name of the problem")
    problem_folder: Path = Field(..., description="The folder of the problem")
    code_path: Optional[Path] = Field(None, description="The path to the code file")
    outfile_path: Optional[Path] = Field(None, description="The path of the output file")

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.code_path is None:
            self.code_path = self.problem_folder / f"{self.problem_name}_generated.py"
        if self.outfile_path is None:
            self.outfile_path = self.problem_folder / f"{self.problem_name}_generated.out"

    def save_code(self, code_path: Optional[Path]=None) -> Path:
        code_path = self.code_path if code_path is None else code_path
        code_path.write_text(self.source_code)
        return Path(code_path)
    
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