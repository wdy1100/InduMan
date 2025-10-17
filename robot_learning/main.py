""" Launch RL/IL training and evaluation. """

import sys
import signal
import os
import json
import logging
import numpy as np
import torch
from mpi4py import MPI

from robot_learning.config import create_parser
from robot_learning.trainer import Trainer
from robot_learning.utils.logger import Logger
from robot_learning.utils.mpi import mpi_sync

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def run(parser=None):
    """ Runs Trainer. """
    if parser is None:
        parser = create_parser()

    config, unparsed = parser.parse_known_args()
    if len(unparsed):
        Logger.error("Unparsed argument is detected:\n%s", unparsed)
        return

    rank = MPI.COMM_WORLD.Get_rank()
    config.rank = rank
    config.is_chef = rank == 0
    config.num_workers = MPI.COMM_WORLD.Get_size()
    set_log_path(config)

    config.seed = config.seed + rank
    if hasattr(config, "port"):
        config.port = config.port + rank * 2 # training env + evaluation env

    if config.is_chef:
        Logger.warning("Run a base worker.")
        make_log_files(config)
    else:
        Logger.warning("Run worker %d and disable Logger.", config.rank)
        Logger.setLevel(logging.CRITICAL)

    # syncronize all processes
    mpi_sync()

    def shutdown(signal, frame):
        Logger.warning("Received signal %s: exiting", signal)
        sys.exit(128 + signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # set global seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # build a trainer
    trainer = Trainer(config)
    if config.is_train:
        trainer.train()
        Logger.info("Finish training")
    else:
        trainer.evaluate()
        Logger.info("Finish evaluating")

def set_log_path(config):
    """
    Sets paths to log directories.
    """
    config.run_name = "{}.{}.{}.{}".format(
        config.env, config.algo, config.run_prefix, config.seed
    )
    config.log_dir = os.path.join(config.log_root_dir, config.run_name)
    config.record_dir = os.path.join(config.log_dir, "video")
    config.demo_dir = os.path.join(config.log_dir, "demo")

def make_log_files(config):
    """
    Sets up log directories and saves git diff and command line.
    """
    Logger.info("Create log directory: %s", config.log_dir)
    os.makedirs(config.log_dir, exist_ok=config.resume or not config.is_train)

    Logger.info("Create video directory: %s", config.record_dir)
    os.makedirs(config.record_dir, exist_ok=config.resume or not config.is_train)

    Logger.info("Create demo directory: %s", config.demo_dir)
    os.makedirs(config.demo_dir, exist_ok=config.resume or not config.is_train)

    if config.is_train:
        # log config
        param_path = os.path.join(config.log_dir, "params.json")
        Logger.info("Store parameters in %s", param_path)
        with open(param_path, "w") as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)