import logging
import numpy as np
import os
import pandas as pd


def load_npz_data(npz_dir_path, sensors, missing_ok=True):
    logging.info(f"Loading from {npz_dir_path}")
    sensors_in_dir = set([file.replace(r'.npz', '') for file in os.listdir(npz_dir_path) if r'.npz' in file])
    sensors_to_load = sensors_in_dir.intersection(sensors)
    if len(sensors_to_load) != len(sensors):
        logging.info(f"Missing sensors: {set(sensors) - sensors_to_load}")
        if not missing_ok:
            raise FileNotFoundError(repr(set(sensors) - sensors_to_load))
    data = [np.load(os.path.join(npz_dir_path, f"{sensor}.npz")) for sensor in sensors]
    data = np.array([d['arr_0'] for d in data]).T
    df = pd.DataFrame(data=data, columns=sensors)
    return df


def rmse(predictions, targets):
    err = np.sqrt(((predictions - targets) ** 2).mean())
    return round(err, 3)

