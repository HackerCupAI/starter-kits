import os
import base64
import re
import pathlib

def encode_file(file_path):
    """Encode text files or base64 encode image files."""
    if file_path.endswith('.jpg') or file_path.endswith('.gif'):
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
        base64_encoded_data = base64.b64encode(image_data)
        base64_string = base64_encoded_data.decode('utf-8')

        return base64_string
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError as e:
            print(f"Error decoding file {file_path}: {e}")
            return None

def extract_images(markdown_content):
    """Extract PHOTO_IDs from markdown files and return as a list."""
    return re.findall(r'\{\{PHOTO_ID:(\d+)\|WIDTH:\d+\}\}', markdown_content)

def get_problem_files(directory):
    all_file_paths = [os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk(directory) for filename in filenames]

    statement_files = [f for f in all_file_paths if f.endswith('.md') and not f.endswith('sol.md')]
    data = {}
    images_map = {}
    
    for sfile in statement_files:
        original_statement_filepath = sfile
        sfile = sfile.replace(directory, "")
        problem_id = (sfile.split('.')[0]).split('/')[-1]
        if problem_id not in data and problem_id != "":
            content = encode_file(original_statement_filepath)
            data[problem_id] = {
                'problem_file': original_statement_filepath,
                "id": problem_id,
                "problem" : content,
                'input': None,
                'input_file': None,
                'output': None,
                'output_file': None,
                'sample_input': None,
                'sample_input_file': None,
                'sample_output':None,
                'sample_output_file':None,
                'images': []

            }
            image_ids = extract_images(content)
            images_map[problem_id] = image_ids
        else:
            print("Data duplicate:", problem_id)
    
    all_image_files = []

    for file_path in all_file_paths:
        original_filepath = file_path
        file_path = file_path.replace(directory, "")
        file_path = file_path.replace('_sol.', '.')
        file_type = file_path.split('.')[-1]
        problem_id = (file_path.split('.')[0]).split('/')[-1]
        problem_id = problem_id.replace("_sample_input", "")
        problem_id = problem_id.replace("_sample_output", "")
        if file_type in ['html', 'jpg', 'gif']:
            all_image_files.append(original_filepath)
        if problem_id in data:
            content = encode_file(original_filepath)
            if file_type == 'in':
                data[problem_id]['input_file'] = original_filepath
                data[problem_id]['input'] = content
            elif file_type == 'out':
                data[problem_id]['output_file'] = original_filepath
                data[problem_id]['output'] = content
            elif file_type == 'md':
                    data[problem_id]['problem_file'] = original_filepath
                    data[problem_id]['problem'] = content
            elif file_type == 'txt':
                if "sample_input" in file_path:
                    data[problem_id]["sample_input_file"] = original_filepath
                    data[problem_id]["sample_input"] = content
                if "sample_output" in file_path:
                    data[problem_id]["sample_output_file"] = original_filepath
                    data[problem_id]["sample_output"] = content
        else:
            # print("output not recognized", original_filepath)
            continue


        for key in list(images_map.keys()):
            photo_ids = images_map[key]
            if len(photo_ids) > 0:
                image_result = []
                for photo_id in photo_ids:
                    ids_found = [s for s in all_image_files if f"{photo_id}.jpg" in s]
                    if len(ids_found) == 1:
                        image_file = ids_found[0]
                        image_content = encode_file(image_file)
                        item = {"filepath": image_file, "base64_content": image_content, "id": photo_id}
                        image_result.append(item)
                data[key]['images'] = image_result
            if len(image_result) > 0:
                md =  data[key]["problem"] 
                for image_object in image_result:
                    id = image_object["id"]
                    img_id = f"PHOTO_ID:{id}"
                    img_file = image_object["filepath"]
                    md = md.replace(img_id, f"<img {img_file}>")
                data[key]["problem"] = md
                   
    return data

       
def get_file_contents(filename):
    content = None
    with open(f"{filename}", "r") as f:
        content = f.read()
    return content

def add_samples(markdown_content, input_sample_file, output_sample_file):
    markdown_content+=f"\nInput samples: {{<txt {input_sample_file}>}}\nOutput samples: {{<txt {output_sample_file} >}}" 
    return markdown_content

def add_samples_to_markdown(problemset_files):
    for problem_id in list(problemset_files.keys()):
        markdown = problemset_files[problem_id]["problem"]
        input_sample_file = problemset_files[problem_id]["sample_input_file"]
        output_sample_file = problemset_files[problem_id]["sample_output_file"]
        md = add_samples(markdown, input_sample_file, output_sample_file)
        problemset_files[problem_id]["problem"] = md
        return problemset_files


def get_problemset(directory_path):
    problemset_files = get_problem_files(directory_path)
    problemset = add_samples_to_markdown(problemset_files)
    return problemset


def mkdirp(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)