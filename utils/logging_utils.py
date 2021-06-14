import logging
import sys
from clearml import Task


def register_logger(log_file=None, stdout=True):
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    handlers = []

    if stdout:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(format="%(asctime)s %(message)s",
                        handlers=handlers,
                        level=logging.INFO,
                        )
    logging.root.setLevel(logging.INFO)


def log_args_description(args):
    """
    Logs the content of the arguments
    Args:
        args: instance of arguments object, i.e. parser.parse_args()
    """
    args_header = """
    ====== All settings used ======:\n
"""

    s = ""
    for k, v in sorted(vars(args).items()):
        s += f"      {k}: {v}\n"

    logging.info(args_header + s)


def get_clearml_logger(project_name, task_name, tags=None):
    task = Task.init(project_name=project_name, task_name=task_name, tags=tags)
    logger = task.get_logger()
    return logger


def log_command():
    logging.info("Running command:")
    logging.info(' '.join(sys.argv))
