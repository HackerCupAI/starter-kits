import base64
import re
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Optional, List

def _encode_image(image_path):
    with open(image_path, "rb") as image_file:
        img = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{img}"

def _find_used_images(description_text: str, folder_path: Path) -> list[Path]:
    all_images = list(folder_path.glob('*.jpg'))
    
    photo_ids = set(re.findall(r'{{PHOTO_ID:(\d+)', description_text))
    markdown_images = set(re.findall(r'!\[.*?\]\((.*?\.jpg)\)', description_text))
    
    used_images = [
        img for img in all_images 
        if img.stem in photo_ids or img.name in markdown_images
    ]
    
    return used_images

def _replace_img_links(description_text: str, image_paths: list[Path]) -> str:
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
    input: str
    output: str
    folder_path: Path
    code: Optional[str] = None
    images: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._process_description_and_images()

    def _process_description_and_images(self):
        used_images = _find_used_images(self.problem_description, self.folder_path)
        self.problem_description = _replace_img_links(self.problem_description, used_images)
        self.images = [_encode_image(str(image_path)) for image_path in used_images]

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
            input=input_path.read_text(),
            output=input_path.with_suffix('.out').read_text(),
            folder_path=input_path.parent,
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
            folder_path=folder_path,
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
        logging.info(f"Found {len(problems)} problems in folder: {folder_path}")
        return problems
    
    def exec(self, code: Optional[str] = None, input: Optional[str] = None, timeout: int = 10):
        logging.info("Running solution synchronously...")
        if code is None:
            code = self.code
        if input is None:
            input = self.input

        vars = {}
        try:
            exec(code, vars)
        except Exception as e:
            logging.error(f"The generated code is not valid: {code}")
            return None
        try:
            fn = vars.get("solve", lambda x: x)
            return fn(input)
        except Exception as e:
            logging.error(f"Error executing code: {e}")
            return None


    def __repr__(self):
        return f"""Problem: {self.name}
    Description: {self.problem_description[:50]}...
    Sample Input: {self.sample_input[:50]}...
    Sample Output: {self.sample_output[:50]}...
    Input: {self.input[:50]}...
    Images: {len(self.images)} image(s)
"""


if __name__ == "__main__":
    problem_name ="cheeseburger_corollary_ch1"
    folder_path = Path("../dataset/2023/practice/")

    # load 1 problem by name
    problem = Problem.from_name(
        problem_name, folder_path
    )
    logging.info(problem)

    # load all problems in folder
    problems = Problem.find_all(folder_path)
    logging.info(problems)