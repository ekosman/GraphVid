import argparse
import logging

args_header = """
====== All settings used ======:\n
"""


def log_args_description(args):
    """
    Logs the content of the arguments
    Args:
        args: instance of arguments object, i.e. parser.parse_args()
    """
    s = ""
    for k, v in sorted(vars(args).items()):
        s += f"      {k}: {v}\n"

    logging.info(args_header + s)


def get_default_training_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_path',
                        default=f"/home/koe1tv/data_2wp/Idiada",
                        help="path to directory containing subdirectory with npz sensors data")
    parser.add_argument('--log_every', type=int, default=100, help="log interval during training")
    parser.add_argument('--save_every', type=int, default=100, help="model checkpoint interval during training")
    parser.add_argument('--epochs', type=int, default=1000, help="number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=1000, help="batch size for training")
    parser.add_argument('--exps_dir',
                        default=r"/mnt/internal_2tb/exps/anomaly_detection_all",
                        help="where to save all the outputs: models, visualizations, log, tensorboard")
    return parser


def update_args(args, configs):
    for k, v in configs.items():
        setattr(args, k, v)
