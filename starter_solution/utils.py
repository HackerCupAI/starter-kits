import base64
import glob
import os
import re
import simple_parsing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def find_used_images(description_text: str, folder_path: Path) -> list[Path]:
    all_images = list(folder_path.glob('*.jpg'))
    
    photo_ids = set(re.findall(r'{{PHOTO_ID:(\d+)', description_text))
    markdown_images = set(re.findall(r'!\[.*?\]\((.*?\.jpg)\)', description_text))
    
    used_images = [
        img for img in all_images 
        if img.stem in photo_ids or img.name in markdown_images
    ]
    
    return used_images

def replace_img_links(description_text: str, image_paths: list[Path]) -> str:
    for image_path in image_paths:
        image_id = image_path.stem
        old_ref = f"{{{{PHOTO_ID:{image_id}|WIDTH:600}}}}"
        new_ref = f"![{image_id}]({image_path.name})"
        description_text = description_text.replace(old_ref, new_ref)
    
    return description_text

@dataclass
class Problem:
    name: str
    problem_description: str
    sample_input: str
    sample_output: str
    input_path: Path
    output_path: Path
    images: list[str] = field(default_factory=list)
    _folder_path: Path = field(init=False, repr=False)

    def __post_init__(self):
        self._folder_path = self.input_path.parent
        self._process_description_and_images()

    def _process_description_and_images(self):
        used_images = find_used_images(self.problem_description, self._folder_path)
        self.problem_description = replace_img_links(self.problem_description, used_images)
        self.images = [encode_image(str(image_path)) for image_path in used_images]

    @classmethod
    def from_name(cls, name: str, folder_path: Path):
        description_path = folder_path / f"{name}.md"
        input_path = folder_path / f"{name}.in"
        output_path = folder_path / f"{name}.out"
        sample_input_path = folder_path / f"{name}_sample_input.txt"
        sample_output_path = folder_path / f"{name}_sample_output.txt"

        return cls.from_files(
            name=name,
            description_path=description_path,
            sample_input_path=sample_input_path,
            sample_output_path=sample_output_path,
            input_path=input_path,
        )

    @classmethod
    def from_files(cls, name: str, description_path: Path, sample_input_path: Path, 
                   sample_output_path: Path, input_path: Path):
        return cls(
            name=name,
            problem_description=description_path.read_text(),
            sample_input=sample_input_path.read_text(),
            sample_output=sample_output_path.read_text(),
            input_path=input_path,
            output_path=input_path.with_suffix('.out'),
        )

    @classmethod
    def from_folder(cls, folder_path: Path):
        sample_input_path = next(folder_path.glob('*_samples.in'))
        sample_output_path = next(folder_path.glob('*_samples.out'))
        description_path = next(folder_path.glob('*.md'))
        name = description_path.stem
        input_path = next(f for f in folder_path.glob('*.in') if not f.name.endswith('_samples.in'))

        return cls.from_files(
            name=name,
            description_path=description_path,
            sample_input_path=sample_input_path,
            sample_output_path=sample_output_path,
            input_path=input_path,
        )

    @classmethod
    def find_all(cls, folder_path: Path) -> List['Problem']:
        problems = []
        
        # Find all markdown files in the folder
        md_files = folder_path.glob('*.md')
        
        for md_file in md_files:
            # Skip files that end with '_sol.md' as they might be solution files
            if md_file.stem.endswith('_sol'):
                continue
            
            problem_name = md_file.stem
            try:
                problem = cls.from_name(problem_name, folder_path)
                problems.append(problem)
            except FileNotFoundError as e:
                print(f"Warning: Couldn't create problem from {problem_name}. Error: {e}")
        print(f"Found {len(problems)} problems in folder: {folder_path}")
        return problems

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
        solution = solution[len("```python") :]
    if solution.endswith("```"):
        solution = solution[: -len("```")]
    return solution


class ProblemArgs(simple_parsing.Serializable):
    problem_name: str = "cheeseburger_corollary_ch1"
    folder_path: Path = Path("dataset/2023/practice/")

if __name__ == "__main__":
    args = simple_parsing.parse(ProblemArgs)
    problem = Problem.from_name(
        args.problem_name, args.folder_path
    )
    print(problem)

    problems = Problem.find_all(args.folder_path)
    print(problems)