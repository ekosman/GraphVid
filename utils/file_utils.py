import os
import shutil
from pathlib import Path
from os import path


def create_folder_safe(full_path_and_file):
    path = (os.path.dirname(full_path_and_file))
    Path(path).mkdir(parents=True, exist_ok=True)


def save_str(full_path_and_file, string):
    create_folder_safe(full_path_and_file)
    text_file = open(full_path_and_file, "w")
    text_file.write(string)
    text_file.close()


def load_string(full_path_and_file):
    text_file = open(full_path_and_file, "r")
    res = text_file.read()
    text_file.close()
    return res


def remove_folder_recursively(path):
    # check if dir exists
    if not os.path.isdir(path):
        return
    shutil.rmtree(path)


def file_exists(full_path):
    return path.exists(full_path)