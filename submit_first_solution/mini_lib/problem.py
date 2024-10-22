import base64
import re
from pydantic import BaseModel, Field
import logging
from pathlib import Path
from typing import Optional, Any

from .utils import _name_to_snake_case

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

class Problem(BaseModel):
    folder_path: Path = Field(..., description="The path to the problem directory")
    name: str = Field(..., description="The name of the problem")
    problem_description: str = Field(..., description="The description of the problem")
    sample_input: Path = Field(..., description="The path to the sample input of the problem")
    sample_output: Path = Field(..., description="The path to the sample output of the problem")
    input_path: Path = Field(..., description="The path to the input file")
    output_path: Path = Field(..., description="The path to the output file")
    code: Optional[str] = None
    images: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self._process_description_and_images()
        self.name = _name_to_snake_case(self.name)

    def _process_description_and_images(self):
        used_images = _find_used_images(self.problem_description, self.folder_path)
        self.problem_description = _replace_img_links(self.problem_description, used_images)
        self.images = [_encode_image(str(image_path)) for image_path in used_images]

    def get_sample_input(self) -> str:
        return self.sample_input.read_text()

    def get_sample_output(self) -> str:
        return self.sample_output.read_text()

    def get_input(self) -> str:
        return self.input_path.read_text()

    def get_output(self) -> str:
        return self.output_path.read_text()

    @classmethod
    def from_name(cls, name: str, base_path: Path):
        # Detect if we're using the new folder structure
        new_structure_path = base_path / name
        if new_structure_path.is_dir():
            # New folder-naming based structure
            description_path = new_structure_path / "statement.txt"
            input_path = new_structure_path / "full_in.txt"
            output_path = new_structure_path / "full_out.txt"
            sample_input_path = new_structure_path / "sample_in.txt"
            sample_output_path = new_structure_path / "sample_out.txt"
            folder_path = new_structure_path
        else:
            # Original flat file structure
            description_path = base_path / f"{name}.md"
            input_path = base_path / f"{name}.in"
            output_path = base_path / f"{name}.out"
            sample_input_path = base_path / f"{name}_sample_input.txt"
            sample_output_path = base_path / f"{name}_sample_output.txt"
            folder_path = base_path

        return cls.from_files(
            name=name,
            description_path=description_path,
            sample_input_path=sample_input_path,
            sample_output_path=sample_output_path,
            input_path=input_path,
            output_path=output_path,
            folder_path=folder_path,
        )
    
    def load_code(self) -> Path:
        "Guess and load the code file for the problem"
        is_2024_problem = _name_to_snake_case(self.folder_path.stem) == self.name
        if is_2024_problem:
            for file in self.folder_path.glob("*.cpp"):
                return file
            for file in self.folder_path.glob("*.py"):
                return file
        else:
            if (self.folder_path / f"{self.name}.cpp").exists():
                return self.folder_path / f"{self.name}.cpp"
            elif (self.folder_path / f"{self.name}.py").exists():
                return self.folder_path / f"{self.name}.py"
            else:
                raise ValueError(f"No code file found for problem {self.name}")


    @classmethod
    def from_files(cls, name: str, description_path: Path, sample_input_path: Path, 
                   sample_output_path: Path, input_path: Path, output_path: Path = None,
                   folder_path: Path = None):
        return cls(
            name=name,
            problem_description=description_path.read_text(),
            sample_input=sample_input_path,
            sample_output=sample_output_path,
            input_path=input_path,
            output_path=output_path if output_path else input_path.with_suffix('.out'),
            folder_path=folder_path if folder_path else input_path.parent,
        )

    def __str__(self):
        return (
            f"Problem: {self.name}\n"
            f"Description: {self.problem_description[:50]}...\n"
            f"Sample Input: {self.sample_input.name}\n"
            f"Sample Output: {self.sample_output.name}\n"
            f"Input Path: {self.input_path.name}\n"
            f"Output Path: {self.output_path.name}\n"
            f"Images: {len(self.images)} image(s)\n"
        )
    
def find_problems(base_path: Path) -> list[Problem]:
    """
    Find all the problems in the given base path, supporting both old and new folder structures.
    """
    problems = []

    # Check if base_path contains problem directories (new structure)
    has_problem_dirs = any(entry.is_dir() for entry in base_path.iterdir())
    if has_problem_dirs:
        # New folder-naming based structure
        for problem_dir in base_path.iterdir():
            if problem_dir.is_dir():
                problem_name = problem_dir.name
                try:
                    problem = Problem.from_name(problem_name, base_path)
                    problems.append(problem)
                except Exception as e:
                    logging.error(f"Error loading problem '{problem_name}': {e}")
    else:
        # Original flat file structure
        problem_files = list(base_path.glob("*.in"))
        problem_names = [f.stem for f in problem_files]
        for problem_name in problem_names:
            try:
                problem = Problem.from_name(problem_name, base_path)
                problems.append(problem)
            except Exception as e:
                logging.error(f"Error loading problem '{problem_name}': {e}")

    logging.info(f"Found {len(problems)} problems")
    return problems


if __name__ == "__main__":
    problem_name ="cheeseburger_corollary_ch1"
    folder_path = Path("../dataset/2023/practice/")

    # load 1 problem by name
    problem = Problem.from_name(
        problem_name, folder_path
    )
    print(problem)


    # load all problems in folder
    folder_path = Path("../dataset/2023/practice")
    problems = find_problems(folder_path)
    print(f"Found {len(problems)} problems in folder: {folder_path}")
    assert len(problems) == 5, f"Expected 5 problems, got {len(problems)}"

    # load all problems in folder
    folder_path = Path("../dataset/2024/practice")
    problems = find_problems(folder_path)
    print(f"Found {len(problems)} problems in folder: {folder_path}")
    assert len(problems) == 5, f"Expected 5 problems, got {len(problems)}"
