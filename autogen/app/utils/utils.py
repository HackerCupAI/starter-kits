import os
import base64
import re

def get_problem_files(directory):
    problem_output = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has .md extension
        if filename.endswith(".md"):
            problem_output["problem"] = directory+"/"+ filename
        elif filename.endswith("samples.in"):
            problem_output['input_samples'] = directory+"/"+filename
        elif filename.endswith(".in"):
            problem_output["input"] = directory+"/"+filename
        elif filename.endswith('samples.out'):
            problem_output['output_samples'] = directory+"/"+filename
        elif filename.endswith(".out"):
            problem_output["output"] = directory+"/"+filename

    return problem_output


def parse_markdown(directory_path, problemset_files):
    md_file = problemset_files["problem"]
    with open(f"{md_file}", "r") as f:
        markdown_content = f.read()
        pattern = r"PHOTO_ID:\d+"
        matches = re.findall(pattern, markdown_content)
        print(matches)
        img_matches = {}
        for photo_id in matches:
            id = photo_id.replace("PHOTO_ID:", "")
            if len(id) > 0:
                image_file = f"{directory_path}/{id}.jpg"
                markdown_content = markdown_content.replace(photo_id, f"<img {image_file}>")
        return (markdown_content, img_matches)
        
def get_file_contents(filename):
    content = None
    with open(f"{filename}", "r") as f:
        content = f.read()
    return content

def add_samples(markdown_content, input_sample_file, output_sample_file):
    markdown_content+=f"\nInput samples: {{<txt {input_sample_file}>}}\nOutput samples: {{<txt {output_sample_file}}}" 
    return markdown_content


def get_problemset(directory_path):
    problemset_files = get_problem_files(directory_path)
    (problem_statement, img_matches) = parse_markdown(directory_path, problemset_files)
    # input = get_file_contents(problemset_files["input"])
    # output = get_file_contents(problemset_files["output"])
    input_samples  = get_file_contents(problemset_files["input_samples"])
    output_samples = get_file_contents( problemset_files['output_samples'])
    problem_statement = add_samples( problem_statement, problemset_files['input_samples'],problemset_files['output_samples'] )
    return ({"type": "problem", "contents": problem_statement, "location": problemset_files["problem"]}, 
            {"type": "input",  "location": problemset_files["input"]},
            {"type": "output", "location": problemset_files["output"]},
            {"type": "input_samples", "contents": input_samples, "location":problemset_files['input_samples']},
            {"type": "output_samples", "contents": output_samples, 'location':problemset_files['output_samples']},
            {"type": "images", "contents": img_matches}     
            )