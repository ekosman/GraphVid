import argparse
import logging
import warnings
from os import path

import numpy as np
import torch
from torch_geometric.data import DataLoader, DataListLoader
from tqdm import tqdm

from loaders.kinetics_loader import Kinetics
from network.model_utils import get_model
from utils.PackageUtils.ArgparseUtils import update_args, log_args_description
from utils.PackageUtils.DateUtils import get_time_str
from utils.PackageUtils.FileUtils import load_json
from utils.PackageUtils.TorchUtils import TorchModel
from utils.PackageUtils.logging import OutputDirs
from utils.transforms_utils import build_transforms

warnings.filterwarnings("ignore")


def classification_batch_splitter(batch):
    """
    Creates a corresponding input-output match for classification training
    :param batch: input data for AutoEncoder
    :return: tuple (inputs, targets)
    """
    return (batch[0],), batch[1]


def test_video_recognition(
        output_dirs=None,
        model_path=None,
        args=None,
        **kwargs,
):
    use_data_parallel = False

    loader = DataLoader if not use_data_parallel else DataListLoader

    test_loader = Kinetics(
        dataset_path=args.dataset_path_test,
        transform=build_transforms(superpixels=args.superpixels, train=False),
        # cache_root=f'/media/eitank/disk2T/Datasets/kinetics400/{args.superpixels}/cache/test',
        return_name=True,
        **vars(args),
    )

    test_iter = loader(test_loader,
                       batch_size=args.batch_size,
                       shuffle=False,
                       num_workers=args.num_workers,
                       pin_memory=True)

    # Create the model
    if args.model_path is not None:
        model = TorchModel.load_model(model_path)
    else:
        model = get_model(num_features=3, num_classes=test_loader.num_classes, arch=args.model_type)

    model.to(args.device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print(f"Model's number of parameters: {pytorch_total_params}")

    # criterion = LossWrapper(CrossEntropyLoss())
    model = model.eval()
    video_predictions = dict()
    video_labels = dict()
    with torch.no_grad():
        for data, labels, video_names in tqdm(test_iter):
            outputs = model(data.to(args.device)).softmax(dim=-1).cpu()

            for output, label, video_name in zip(outputs, labels, video_names):
                video_name = video_name.item()
                if video_name not in video_predictions:
                    video_predictions[video_name] = output
                else:
                    video_predictions[video_name] = torch.vstack([video_predictions[video_name], output])

                video_labels[video_name] = label

    for num_views in range(2, 20):
        n_true_top1 = 0
        n_true_top5 = 0
        for video_name, _ in video_predictions.items():
            pred = video_predictions[video_name].view(-1, 400)
            sample_idx = np.random.choice(len(pred), size=min(num_views, len(pred)), replace=False)


            pred = pred[sample_idx].mean(dim=0)
            top1_indices = pred.topk(1).indices
            top5_indices = pred.topk(5).indices
            y_true = torch.tensor(video_labels[video_name])
            n_true_top1 += (top1_indices == y_true.view(1).expand_as(top1_indices)).sum()
            n_true_top5 += (top5_indices == y_true.view(1).expand_as(top5_indices)).sum()

        print(f"Accuracy ({num_views} views) (top  1): {n_true_top1 / len(video_labels)} \t|\t (top 5) {n_true_top5 / len(video_predictions)}")


def get_args():
    parser = argparse.ArgumentParser(description="Train signals prediction")
    # io
    parser.add_argument('--dataset_path_test', help='path to a directory containing all the data')
    parser.add_argument('--log_every', default=1, type=int, help='logging intervals while training (iterations)')
    parser.add_argument('--num_workers', default=7, type=int, help='')  # 7
    parser.add_argument('--num_views', default=4, type=int, help='')
    parser.add_argument('--steps_between_frames', default=1, type=int, help=r'')
    parser.add_argument('--step_between_clips', default=1, type=int, help=r'')
    parser.add_argument('--frames_per_clip', default=16, type=int, help=r'')
    parser.add_argument('--model_type',
                        default='gcn',
                        # choices=['gcn', 'gat', 'simple_gcn', 'pna'],
                        type=str,
                        help='which model to use for training')
    parser.add_argument('--model_path',
                        type=str,
                        help=r'path of a pickled torch model to load')
    parser.add_argument('--task_name',
                        default='no_name',
                        type=str,
                        help=r'name of the task. used as namespace for saving output directory')
    parser.add_argument('--exps_dir',
                        # default=r"./exps",
                        help="where to save all the outputs: models, visualizations, log, tensorboard")
    parser.add_argument('--batch_size', type=int, default=800, help="batch size for training")
    parser.add_argument('--superpixels', type=int, default=50, help="number of superpixels")
    parser.add_argument('--device', type=str, default='cuda', help="device to use for inference of torch models")
    # parser.add_argument('--dataset', type=str, default='comma', help="which loader to use",
    #                     choices=['comma', 'udacity'])
    parser.add_argument('--show_errors',
                        default=False,
                        action='store_true',
                        help="show all errors from the loader")
    parser.add_argument('--disable_clearml_logger',
                        default=False,
                        action='store_true',
                        help="disable logging to clearml server")
    parser.add_argument('--config_file',
                        type=str,
                        help='path to a json files containing further configurations for this script. Good for model-specific configurations')

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()
    configs = load_json(args.config_file)
    update_args(args, configs)

    time_str = get_time_str()

    exps_dir = path.join(args.exps_dir, args.task_name, time_str)
    output_dirs = OutputDirs(exps_dir).register_dirs().register_logger().log()

    log_args_description(args)

    logging.info(f"Using device {args.device}")

    test_video_recognition(
        **vars(args),
        output_dirs=output_dirs,
        args=args
    )
