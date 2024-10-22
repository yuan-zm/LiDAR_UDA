# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys, shutil


def setup_logger(name, cfg, filename="log.txt", distributed_rank=1):
    
    save_dir = cfg.TRAIN.LOG_DIR
    os.makedirs(os.path.join(save_dir, 'files'), exist_ok=True)
    if not cfg.TRAIN.DEBUG:
        shutil.copytree('network', os.path.join(save_dir, 'files', 'network'))
        shutil.copytree('dataset', os.path.join(save_dir, 'files', 'dataset'))
        shutil.copytree('configs', os.path.join(save_dir, 'files', 'configs'))
        shutil.copytree('trainers', os.path.join(save_dir, 'files', 'trainers'))

        shutil.copy('run.sh', os.path.join(save_dir, 'files', 'run.sh'))
        shutil.copy('trainer_base.py', os.path.join(save_dir, 'files', 'trainer_base.py'))
        shutil.copy('run_experiments.py', os.path.join(save_dir, 'files', 'run_experiments.py'))
        shutil.copy(cfg.TRAIN.config_file, os.path.join(save_dir, 'files', cfg.TRAIN.config_file.split('/')[-1]))

    log_fname = os.path.join(save_dir, 'log_train.txt')
    logging.basicConfig(filename=log_fname)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_tb(tb_writer, info_name, info_paras, save_iter):
    for name, para in zip(info_name, info_paras):
        tb_writer.add_scalar(name, para, save_iter)