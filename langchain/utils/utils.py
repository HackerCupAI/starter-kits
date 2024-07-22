import os
import re
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

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

    return problem_output

def parse_markdown(problemset_files,directory_path):
    md_file = problemset_files["problem"]
    with open(f"{md_file}", "r") as f:
        markdown_content = f.read()
        pattern = r"PHOTO_ID:\d+"
        matches = re.findall(pattern, markdown_content)
        for photo_id in matches:
            id = photo_id.replace("PHOTO_ID:", "")
            if len(id) > 0:
                image_file = f"{directory_path}/{id}.jpg"
                markdown_content = markdown_content.replace(photo_id, f"<img {encode_image(image_file)}>")
        return markdown_content
        
def get_file_contents(filename):
    content = None
    with open(f"{filename}", "r") as f:
        content = f.read()
    return content

def add_samples(markdown_content, input_sample_file, output_sample_file):
    markdown_content+=f"\nInput samples: {{<txt {input_sample_file}>}}\nOutput samples: {{<txt {output_sample_file} >}}" 
    return markdown_content

def get_problemset(directory_path):
    problemset_files = get_problem_files(directory_path)
    problem_statement = parse_markdown(problemset_files,directory_path)

    input_samples  = get_file_contents(problemset_files["input_samples"])
    output_samples = get_file_contents( problemset_files['output_samples'])
    problem_statement = add_samples( problem_statement, problemset_files['input_samples'],problemset_files['output_samples'] )
    return ({"type": "problem", "contents": problem_statement, "location": problemset_files["problem"]}, 
            {"type": "input",  "location": get_file_contents(problemset_files["input"])},
            {"type": "input_samples", "contents": input_samples, "location":problemset_files['input_samples']},
            {"type": "output_samples", "contents": output_samples, 'location':problemset_files['output_samples']},    
            )