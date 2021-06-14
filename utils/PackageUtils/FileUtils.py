import json
import os
import shutil
from pathlib import Path
import numpy as np


def path_exists(path):
    return Path(path).exists()


def create_folder_safe(full_path_and_file):
    path = (os.path.dirname(full_path_and_file))
    Path(path).mkdir(parents=True, exist_ok=True)


def save_str(full_path_and_file, string):
    create_folder_safe(full_path_and_file)
    text_file = open(full_path_and_file, "w")
    n = text_file.write(string)
    text_file.close()


# In order to jsonfy numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_dict_as_json(dict0, path):
    json_str = json.dumps(dict0, cls=NumpyEncoder)
    f = open(path, "w")
    f.write(json_str)
    f.close()


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


def load_json(json_path):
    """
    Loads a json file
    Args:
        json_path: Path to the json file
    Returns: The content of the json file
    """
    if json_path is not None and os.path.exists(json_path):
        with open(json_path, 'r') as fp:
            data = json.load(fp)

        return data

    return dict()


def register_dir(dir_path):
    """
    Create a new directory for the given path. If the directory already exists, does nothing
    Args:
        dir_path: The path of the new directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def load_time_sync(time_sync_file):
    """
    Loads a dictionary containing a mapping: [ rider name |==> time shift between video and VideoTime_Corr ]
    Positive values mean that the video is delayed
    Args:
        time_sync_file: Exact path of the json file containing the dictionary

    Returns: Mapping between rider name to time shift values

    """
    if os.path.exists(time_sync_file):
        with open(time_sync_file, 'r') as fp:
            time_sync = json.load(fp)
    else:
        time_sync = dict()

    return time_sync


def get_original_video_path(original_videos_dir, name):
    """
    Retrieves the file path for the first video of a rider
    Args:
        original_videos_dir: path to the root directory containing all videos
        name: name of the rider, including the lap, e.g. CHDE10_SDR02_150930_03

    Returns: Full path of the the first video of that rider

    """
    name = name.replace('_WIVW', '')
    return os.path.join(original_videos_dir, name, f'{name}_GV_A.mp4')
