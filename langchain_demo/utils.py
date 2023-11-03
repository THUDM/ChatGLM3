import os
import yaml


def tool_config_from_file(tool_name, directory="Tool/"):
    """search tool yaml and return json format"""
    for filename in os.listdir(directory):
        if filename.endswith('.yaml') and tool_name in filename:
            file_path = os.path.join(directory, filename)
            with open(file_path, encoding='utf-8') as f:
                return yaml.safe_load(f)
    return None
