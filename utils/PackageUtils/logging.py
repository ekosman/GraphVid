#!/usr/bin/env python

# Copyright (c) 2016 - for information on the respective copyright owner
# see the NOTICE file and/or the repository
# <https://github.com/boschresearch/amira-blender-rendering>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# logging setup

import os
import logging
import sys
from os import path

from DriveUtils.PackageUtils.FileUtils import register_dir

try:
    from verboselogs import VerboseLogger as getLogger
except ImportError:
    # print("FAIL to import verboselogs, consider installing with pip for additional log levels")
    from logging import getLogger

try:
    import coloredlogs
    coloredlogs.DEFAULT_LEVEL_STYLES["notice"]["color"] = 176
    coloredlogs.DEFAULT_LEVEL_STYLES["critical"]["color"] = "white"
    coloredlogs.DEFAULT_LEVEL_STYLES["critical"]["background"] = "red"
except ImportError:
    # print("FAIL to import coloredlogs, consider installing with pip")
    coloredlogs = None

# local logger configuration
__logger_name = "idiada_2WP"
__logger_logdir = os.path.expandvars("$HOME/Desktop")
__logger_filename = os.path.join(__logger_logdir, f"{__logger_name}.log")
# __logger_loglevel = logging.INFO

__basic_format = "{} {} | {}, {}:{} | {}".format(
    "%(asctime)s",
    "%(levelname)s",
    "%(filename)s",
    "%(funcName)s",
    "%(lineno)d",
    "%(message)s",
)

__colored_format = "%(asctime)s %(filename)s %(lineno)d %(levelname)s %(message)s"

__terminal_format = "[{}] {}:{} | {}".format(
    "%(levelname)s",
    "%(filename)s",
    "%(lineno)d",
    "%(message)s",
)


# HINT: adding a (file) handler and then lowering logger level does not work well
def get_logger(level="INFO", fmt=None):
    """This function returns a logger instance.

    If coloredlogs is installed the messages in terminal will have different colors acording to logging level.
    If verboselogs is installed ist supports additional logging levels and color variations.

    Args:
        level (str, optional): logging level. Defaults to "INFO".
        fmt (str, optional): message format template: fields, order, etc. Defaults to None, which uses a "basic_format".

    Returns:
        logging.Logger: a logger instance.
    """
    # create directory of necessary
    if not os.path.exists(__logger_logdir):
        os.mkdir(__logger_logdir)
    try:
        os.remove(__logger_filename)
    except FileNotFoundError:
        pass

    # setup logger once. Note that basicConfig does nothing (as stated in the
    # documentation) if the root logger was already setup. So we can basically
    # re-call it here as often as we want.
    if fmt is None:
        fmt = __basic_format

    logging.basicConfig(level=level, format=fmt, filename=__logger_filename)

    if coloredlogs is not None:
        coloredlogs.install(
            level=level,
            datefmt="%H:%M:%S",  # default includes date
            # default format:
            # %(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s
            fmt=__colored_format,
        )

    logger = getLogger(__logger_name)

    return logger


def _get_level_enum(level):
    if isinstance(level, int):
        return level
    else:
        return getattr(logging, level)


def add_file_handler(logger, filename=__logger_filename, level="DEBUG"):
    """Explicitly force logging to the terminal (in addition to other logging, e..g to file)"""
    # logger level can block stream-handler
    file_logging_level = _get_level_enum(level)
    logger.debug(f"file_logging_level={file_logging_level}")
    logger.debug(f"logger level={logger.level}")
    if logger.level > file_logging_level:
        logger.setLevel(file_logging_level)

    file_handler = logging.FileHandler(filename)
    set_level(file_handler, level=level)

    file_handler.setFormatter(logging.Formatter(__basic_format))
    logger.addHandler(file_handler)


def add_stream_handler(logger, level="DEBUG"):
    """Explicitly force logging to the terminal (in addition to other logging, e..g to file)"""
    # logger level can block stream-handler
    stream_logging_level = _get_level_enum(level)
    logger.debug(f"stream_logging_level={stream_logging_level}")
    logger.debug(f"logger level={logger.level}")
    if logger.level > stream_logging_level:
        logger.setLevel(stream_logging_level)

    stream_handler = logging.StreamHandler()
    set_level(stream_handler, level=level)

    if coloredlogs is not None:
        fmt = __colored_format
    else:
        fmt = __terminal_format
    stream_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(stream_handler)


def set_level(logger, level="debug"):
    """Set the log level of a logger from a string.

    This is useful in combination with command line arguments."""

    if "debug" == level.lower():
        logger.setLevel(logging.DEBUG)
    elif "info" == level.lower():
        logger.setLevel(logging.INFO)
    elif "warning" == level.lower() or "warn" == level.lower():
        logger.setLevel(logging.WARNING)
    elif "error" == level.lower():
        logger.setLevel(logging.ERROR)
    elif "critical" == level.lower():
        logger.setLevel(logging.CRITICAL)
    elif "disable" == level.lower():
        logger.setLevel(logging.CRITICAL + 1)
    else:
        try:
            logger.setLevel(level)
        except ValueError:
            logger.warning('unsupported level "{}"'.format(level))


def register_logger(log_file=None):
    """
    Registers the global logger with an option to add a file handler.
    After registering the logger with this function, simply use the logging module for creating logs. e.g.:
    - logging.info(...)
    - logging.debug(...)
    - etc
    Args:
        log_file: (optional) path to output the log

    """
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if log_file:
        print(f"Logging to {log_file}")
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(format="%(asctime)s %(message)s",
                        handlers=handlers,
                        level=logging.INFO,
                        )
    logging.root.setLevel(logging.INFO)


def create_output_dirs(root_dir):
    network_dir = path.join(root_dir, 'models')
    visualization_dir = path.join(root_dir, 'visualizations')
    tensorboard_dir = path.join(root_dir, 'tensorboard')
    log_file = path.join(root_dir, f"log.log")
    return {
        'network_dir': network_dir,
        'visualization_dir': visualization_dir,
        'tensorboard_dir': tensorboard_dir,
        'log': log_file
    }


class OutputDirs:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.network_dir = path.join(root_dir, 'models')
        self.visualization_dir = path.join(root_dir, 'visualizations')
        self.tensorboard_dir = root_dir
        self.log_file = path.join(root_dir, f"log.log")

    def register_dirs(self):
        register_dir(self.root_dir)
        register_dir(self.network_dir)
        register_dir(self.visualization_dir)
        register_dir(self.tensorboard_dir)
        return self

    def register_logger(self):
        register_logger(self.log_file)
        return self

    def log(self):
        logging.info("Running command:")
        logging.info(' '.join(sys.argv))
        logging.info(f"Saving models to {self.network_dir}")
        logging.info(f"Saving visualizations to {self.visualization_dir}")
        logging.info(f"Saving tenosrboard to {self.tensorboard_dir}")
        return self
