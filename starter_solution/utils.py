from pathlib import Path
from typing import Optional

import base64
from dataclasses import dataclass
import os
import glob

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

@dataclass
class Problem:
    name: str
    problem_description: str
    sample_input: str
    sample_output: str
    input_path: Path
    output_path: Path
    images: list[str]
    

    @classmethod
    def from_folder(cls, folder_path: Path):
        # Find files
        sample_input = next(folder_path.glob('*_samples.in'))
        sample_output = next(folder_path.glob('*_samples.out'))
        description = next(folder_path.glob('*.md'))
        # problem name from the .md file
        name = description.stem

        input_file = next(f for f in folder_path.glob('*.in') if not f.name.endswith('_samples.in'))
        images = [encode_image(str(image_path)) for image_path in folder_path.glob('*.jpg')]

        # Read file contents
        sample_input_content = sample_input.read_text()
        sample_output_content = sample_output.read_text()
        problem_description_content = description.read_text()

        return cls(
            name=name,
            problem_description=problem_description_content,
            sample_input=sample_input_content,
            sample_output=sample_output_content,
            input_path=input_file,
            output_path=input_file.with_suffix('.out'),
            images=images,
        )
    
    def __repr__(self):
        return f"""Problem: {self.name}
    Description: {self.problem_description[:50]}...
    Sample Input: {self.sample_input[:50]}...
    Sample Output: {self.sample_output[:50]}...
    Input: {self.input_path}
    Images: {len(self.images)} image(s)
"""

def maybe_remove_backticks(solution: str) -> str:
    if solution.startswith("```python"):
        solution = solution[len("```python"):]
    if solution.endswith("```"):
        solution = solution[:-len("```")]
    return solution

if __name__ == '__main__':
    sample_path = Path("assets/cheeseburger_corollary_1")
    problem = Problem.from_folder(sample_path)
    print(problem)