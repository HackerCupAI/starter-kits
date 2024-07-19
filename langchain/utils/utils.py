import os
import re
directory_path = "./assets/practice_problem"  

def get_problem_files(directory):
    problem_output = {}
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has .md extension
        if filename.endswith(".md"):
            problem_output["problem"] = filename
        elif filename.endswith(".in"):
            problem_output["input"] = filename
        elif filename.endswith(".out"):
            problem_output["output"] = filename

    return problem_output

def parse_markdown(problemset_files):
    md_file = problemset_files["problem"]
    with open(f"{directory_path}/{md_file}", "r") as f:
        markdown_content = f.read()
        pattern = r"PHOTO_ID:\d+"
        matches = re.findall(pattern, markdown_content)
        for photo_id in matches:
            id = photo_id.replace("PHOTO_ID:", "")
            if len(id) > 0:
                image_file = f"{directory_path}/{id}.jpg"
                markdown_content = markdown_content.replace(photo_id, f"<img {image_file}>")
        return markdown_content
        
def get_file_contents(filename):
    content = None
    with open(f"{directory_path}/{filename}", "r") as f:
        content = f.read()
    return content

def get_problemset():
    problemset_files = get_problem_files(directory_path)
    problem_statement = parse_markdown(problemset_files)
    input = get_file_contents(problemset_files["input"])
    return ({"type": "problem", "contents": problem_statement, "location": problemset_files["problem"]}, 
            {"type": "input", "contents": input, "location": problemset_files["input"]})