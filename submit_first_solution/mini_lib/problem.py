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
    input_path: Path # this is sometimes a big file
    output_path: Path # this is sometimes a big file
    folder_path: Path
    code: Optional[str] = None
    images: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._process_description_and_images()

    def _process_description_and_images(self):
        used_images = _find_used_images(self.problem_description, self.folder_path)
        self.problem_description = _replace_img_links(self.problem_description, used_images)
        self.images = [_encode_image(str(image_path)) for image_path in used_images]

    def get_input(self) -> str:
        return self.input_path.read_text()

    def get_output(self) -> str:
        return self.output_path.read_text()

    def save_code(self, code: str, code_path: Optional[str] = None, outfile_name: Optional[str] = None):
        final_code = f"from pathlib import Path\ninput = Path('./{self.input_path.name}').read_text()\n\n"
        code_name = f"{self.name}_generated.py"
        code_path = Path(self.folder_path) / code_name if code_path is None else code_path
        final_code += code
        outfile_name = f"./{self.name}_generated.out" if outfile_name is None else outfile_name
        final_code += f"\n\noutput = solve(input)\nPath('{outfile_name}').write_text(output)\n"
        code_path.write_text(final_code)
        return Path(code_path)

    def save_output(self, output: str, outfile: Optional[str] = None):
        outfile_name = f"{self.name}_generated.out"
        outfile = Path(self.folder_path) / outfile_name if outfile is None else outfile
        outfile.write_text(output)
        return Path(outfile)

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
            folder_path=input_path.parent,
        )

    @classmethod
    def find_all(cls, folder_path: Path) -> List['Problem']:
        problems = []
        
        # Find all markdown files in the folder
        md_files = folder_path.rglob('*.md')
        
        for md_file in md_files:
            # Skip files that end with '_sol.md' as they might be solution files
            if md_file.stem.endswith('_sol'):
                continue
            
            problem_name = md_file.stem
            try:
                problem = cls.from_name(problem_name, md_file.parent)
                problems.append(problem)
            except FileNotFoundError as e:
                print(f"Warning: Couldn't create problem from {problem_name}. Error: {e}")
        logging.info(f"Found {len(problems)} problems in folder: {folder_path}")
        return problems


    def __repr__(self):
        return f"""Problem: {self.name}
    Description: {self.problem_description[:50]}...
    Sample Input: {self.sample_input[:50]}...
    Sample Output: {self.sample_output[:50]}...
    Input Path: {self.input_path}
    Output Path: {self.output_path}
    Images: {len(self.images)} image(s)
"""


if __name__ == "__main__":
    problem_name ="cheeseburger_corollary_ch1"
    folder_path = Path("../dataset/2023/practice/")

    # load 1 problem by name
    problem = Problem.from_name(
        problem_name, folder_path
    )
    print(problem)


    # load all problems in folder
    folder_path = Path("../dataset/2023/")
    problems = Problem.find_all(folder_path)
    print(f"Found {len(problems)} problems in folder: {folder_path}")
    assert len(problems) == 29