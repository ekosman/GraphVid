import argparse
import logging
import warnings
from os import path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from loaders.kinetics_loader import Kinetics
from utils.PackageUtils.ArgparseUtils import update_args, log_args_description
from utils.PackageUtils.DateUtils import get_time_str
from utils.PackageUtils.FileUtils import load_json
from utils.PackageUtils.TorchUtils import TorchModel
from utils.PackageUtils.callback_utils import DefaultModelCallback, TensorBoardCallback
from utils.PackageUtils.logging import OutputDirs
from utils.logging_utils import get_clearml_logger
from utils.transforms_utils import build_transforms

"""

"""


def classification_batch_splitter(batch):
    """
    Creates a corresponding input-output match for classification training
    :param batch: input data for AutoEncoder
    :return: tuple (inputs, targets)
    """
    return batch


def train_video_recognition(
        output_dirs=None,
        learning_rate=0.001,
        model_path=None,
        args=None,
        **kwargs,
):
    train_loader = Kinetics(
        dataset_path=args.dataset_path_train,
        video_transform=build_transforms(args.video_height, args.video_width),
        signals_transforms=None,
        return_video=args.video_weight != 0,
        split_type='train',
        video_clips=args.video_clips_train,
        **vars(args),
    )

    train_iter = torch.utils.data.DataLoader(train_loader,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=8,
                                             pin_memory=True)

    eval_loader = Kinetics(
        dataset_path=args.dataset_path_validation,
        video_transform=build_transforms(args.video_height, args.video_width),
        signals_transforms=None,
        return_video=args.video_weight != 0,
        split_type='validation',
        video_clips=args.video_clips_validation,
        **vars(args),
    )

    eval_iter = torch.utils.data.DataLoader(eval_loader,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=8,
                                            pin_memory=True)

    # Create the model
    if model_path is None or not path.exists(model_path):
        model = get_model(**vars(args))
        model = TorchModel(model)
    else:
        model = TorchModel.load_model(model_path)

    model.to(args.device)
    logging.info(summary(model, train_loader.get_loader_shape()))
    model.data_parallel()

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tb_writer = SummaryWriter(log_dir=output_dirs.tensorboard_dir)

    timestamp_names = list(range(args.future_steps,
                                 args.future_steps + args.prediction_length * args.prediction_stride,
                                 args.prediction_stride))

    losses_names = ['total']

    model.register_callback(TensorBoardCallback(tb_writer=tb_writer,
                                                loss_names=losses_names))

    model.register_callback(DefaultModelCallback(log_every=args.log_every,
                                                 save_every=args.save_every,
                                                 loss_names=losses_names))

    if args.epochs != 0:
        model.fit(
            train_iter=train_iter,
            eval_iter=eval_iter,
            test_iter=eval_iter,
            criterion=criterion,
            optimizer=optimizer,
            epochs=args.epochs,
            network_model_path_base=output_dirs.network_dir,
            save_every=args.save_every,
            evaluate_every=args.evaluate_every,
            batch_splitter=classification_batch_splitter,
        )


def get_args():
    parser = argparse.ArgumentParser(description="Train signals prediction")
    # io
    parser.add_argument('--dataset_path_train', help='path to a directory containing all the data')
    parser.add_argument('--dataset_path_validation', help='path to a directory containing all the data')
    parser.add_argument('--dataset_path_test', help='path to a directory containing all the data')
    parser.add_argument('--video_clips_train', type=str, help='(optional) path to the pickled videoClips file')
    parser.add_argument('--video_clips_validation', type=str, help='(optional) path to the pickled videoClips file')
    parser.add_argument('--video_clips_test', type=str, help='(optional) path to the pickled videoClips file')
    parser.add_argument('--evaluate_every',
                        default=10,
                        type=int,
                        help=r'perform evaluation every specified amount of epochs. If the evaluation is expensive, '
                             r'you probably want to choose a high value for this')
    parser.add_argument('--log_every',
                        default=100,
                        type=int,
                        help='logging intervals while training (iterations)')
    parser.add_argument('--save_every',
                        default=10,
                        type=int,
                        help=r'saving model checkpoints every specified amount of epochs')
    parser.add_argument('--model_type',
                        default='gcn',
                        choices=['gcn'],
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
                        default=r"/mnt/sda1/exps/signals_prediction/comma_ai",
                        help="where to save all the outputs: models, visualizations, log, tensorboard")
    parser.add_argument('--epochs', type=int, default=1000, help="number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=50, help="batch size for training")
    parser.add_argument('--device', type=str, default='cuda', help="device to use for inference of torch models")
    parser.add_argument('--dataset', type=str, default='comma', help="which loader to use", choices=['comma', 'udacity'])
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate for the optimizer")
    parser.add_argument('--flip_prob', type=float, default=0.5,
                        help="probability of applying augmentation on the horizontal axis")
    parser.add_argument('--show_errors',
                        default=False,
                        action='store_true',
                        help="show all errors from the loader")
    parser.add_argument('--disable_clearml_logger',
                        default=False,
                        action='store_true',
                        help="disable logging to clearml server")
    parser.add_argument('--config_file',
                        default='/home/koe1tv/idiada2wpdatascience/src/signals_predictions/configs/test_video_mfnet_regression.json',
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

    tags = [
        f'Dataset: Kinetics',
        f'Architecture: {args.model_type}',
    ]
    clearml_logger = None if args.disable_clearml_logger \
        else get_clearml_logger(project_name="GraphVid",
                               task_name=get_time_str(),
                                tags=tags)

    logging.info(f"Using device {args.device}")

    train_video_recognition(
        **vars(args),
        output_dirs=output_dirs,
        args=args
    )
