import json

def load_config(file_path):
    with open(file_path) as file:
        configs = json.load(file)
    return configs