"""
etc functions
"""

import os
import json


def create_folder(path: str):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: Creating directory. {path}")


def get_option_dict_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return result
