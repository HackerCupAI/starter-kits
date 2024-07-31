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
    input_path: str
    images: list[str]
    output: Optional[str]

    @classmethod
    def from_folder(cls, folder_path):
        import os
        import glob

        # Find files
        sample_input = glob.glob(os.path.join(folder_path, '*_samples.in'))[0]
        sample_output = glob.glob(os.path.join(folder_path, '*_samples.out'))[0]
        description = glob.glob(os.path.join(folder_path, '*.md'))[0]
        # problem name from the .md file
        name = Path(description).stem

        input_file = glob.glob(os.path.join(folder_path, '*.in'))
        input_file = [f for f in input_file if not f.endswith('_samples.in')][0]
        images = [encode_image(image_path) for image_path in glob.glob(os.path.join(folder_path, '*.jpg'))]

        # Read file contents
        with open(sample_input, 'r') as f:
            sample_input_content = f.read()
        with open(sample_output, 'r') as f:
            sample_output_content = f.read()
        with open(description, 'r') as f:
            problem_description_content = f.read()

        return cls(
            name=name,
            problem_description=problem_description_content,
            sample_input=sample_input_content,
            sample_output=sample_output_content,
            input_path=input_file,
            images=images,
            output=None
        )
    
    def __repr__(self):
        return f"""Problem: {self.name}
    Description: {self.problem_description[:50]}...
    Sample Input: {self.sample_input[:50]}...
    Sample Output: {self.sample_output[:50]}...
    Input: {self.input_path}
    Images: {len(self.images)} image(s)
"""

if __name__ == '__main__':
    sample_path = Path("assets/cheeseburger_corollary_1")
    problem = Problem.from_folder(sample_path)
    print(problem)