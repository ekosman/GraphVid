import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from loaders.kinetics_loader import Kinetics
from torch_geometric.data import DataLoader
from utils.transforms_utils import build_transforms

if __name__ == "__main__":
    dataset_path = r"/media/koe1tv/disk2T/Datasets/kinetics400/valid"
    df = pd.Series(name='time', index=list(range(200, 2001, 200)))
    n_samples = 10000
    batch_size = 200
    # for num_sp in range(600, 2001, 200):
    num_sp = 1600
    loader = Kinetics(
        dataset_path=dataset_path,
        transform=build_transforms(superpixels=num_sp),
    )
    _iter = DataLoader(loader, shuffle=True, batch_size=batch_size, num_workers=7, pin_memory=True)

    calc = 0
    start_time = time.time()
    for x in _iter:
        calc += batch_size
        if calc >= n_samples:
            break

        print(f"SP: {num_sp} | count: {calc} / {n_samples} | {(time.time() - start_time) / calc} sec per sample")

    tot = time.time() - start_time
    df.loc[num_sp] = tot / calc

    df.to_csv("sp_times.csv")
