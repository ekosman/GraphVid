# GraphVid
This repository includes partial documentations for the code for reproducing the results for our paper "GraphVid: It Only Takes a Few Nodes o Understand a Video"

![intro](figures/intro.png)

## Instructions

First, download the kinetics-400 dataset using the code from the following repository (instructions are included):

https://anonymous.4open.science/r/KineticsDownloader-FF74/

Next, use the downloaded dataset to train your model by:

```cmd
python train_kinetics.py --dataset_path_train <path to train set> --dataset_path_validation <path to validation set> --dataset_path_test <path to train set> 
```