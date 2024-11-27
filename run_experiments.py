# Common
import os
import os.path as osp
import datetime
import argparse
import warnings
import socket

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import importlib

import wandb

from network.minkUnet import MinkUNet34

from utils import common as com
from utils.logger_FADA import setup_logger

# config file
from configs.config_base import cfg_from_yaml_file
from easydict import EasyDict

warnings.filterwarnings("ignore")
from git import Repo 

def change_Config_DEBUG(cfg):
    cfg.TRAIN.T_VAL_ITER = cfg.DEBUG.T_VAL_ITER
    cfg.TRAIN.S_VAL_ITER = cfg.DEBUG.S_VAL_ITER
    cfg.TRAIN.LOG_PERIOD = cfg.DEBUG.LOG_PERIOD
    cfg.TRAIN.PREHEAT_STEPS = cfg.DEBUG.PREHEAT_STEPS
    cfg.TRAIN.EXP_NAME = cfg.DEBUG.EXP_NAME

    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='PGDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/SynLiDAR2SemanticKITTI/CosMix.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    args = parser.parse_args()
    return args

def main():
    com.make_reproducible() # freeze all seeds

    args = parse_args()
    # load the configuration
    cfg = EasyDict()
    cfg.OUTPUT_DIR = './workspace/'
    cfg_from_yaml_file(args.config_file, cfg)

    cfg.TRAIN.config_file = args.config_file
    curPath = os.path.abspath(os.path.dirname(__file__))
    cfg.TRAIN.CURPATH = curPath

    repo = Repo(curPath)
    print(repo.active_branch)   # current branch

    wb_note = '*Path: ' + str(curPath)  + '      **Git branch: ' + str(repo.active_branch)

    # set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.TRAIN.GPU_ID)
   
    # mkdir for logs and checkpoints
    time_now = datetime.datetime.now().strftime("%m-%d-%H_%M")
    
    # Init wandb and logger
    if cfg.TRAIN.DEBUG:
        os.environ['WANDB_MODE'] = 'dryrun'
        cfg = change_Config_DEBUG(cfg)
    
    # Init save floder
    cfg.TRAIN.MODEL_DIR = osp.join(cfg.OUTPUT_DIR, cfg.TRAIN.PROJECT_NAME, 'checkpoints', cfg.TRAIN.EXP_NAME, time_now)
    os.makedirs(cfg.TRAIN.MODEL_DIR, exist_ok=True)
    cfg.TRAIN.LOG_DIR = osp.join(cfg.OUTPUT_DIR, cfg.TRAIN.PROJECT_NAME, 'logs', cfg.TRAIN.EXP_NAME, time_now)
    os.makedirs(cfg.TRAIN.LOG_DIR, exist_ok=True)
    cfg.TRAIN.TB_DIR = osp.join(cfg.OUTPUT_DIR, cfg.TRAIN.PROJECT_NAME, 'tb_dirs', cfg.TRAIN.EXP_NAME, time_now)
   
    hostname = socket.gethostname()
    cfg.TRAIN.HOSTNAME = hostname
    cfg.TRAIN.WANDB_ID = str(wandb.util.generate_id())
    
    # WANDB initialization
    wandb.init(name=cfg.TRAIN.EXP_NAME, notes=wb_note,
               project=cfg.TRAIN.PROJECT_NAME,
               entity='zhiminyuan', id=cfg.TRAIN.WANDB_ID)
    wandb.config.update(cfg, allow_val_change=True)

    print(cfg)
   
    # Init logger and tensorboard
    logger = setup_logger("Trainer", cfg)  # Init Logging
    logger.info('this Experiment is: \n %s \n \n ' % wb_note)
    tf_writer = SummaryWriter(cfg.TRAIN.TB_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Host: {}, GPU: {}, wandb_ID: {}".format(hostname, cfg.TRAIN.GPU_ID, cfg.TRAIN.WANDB_ID))

    # init network G and D
    net = MinkUNet34(cfg.MODEL_G).to(device)
    if cfg.OPTIMIZER.TYPE == "Adam":
        G_optim = optim.Adam(net.parameters(), 
                             lr=cfg.OPTIMIZER.LEARNING_RATE_G)
    elif cfg.OPTIMIZER.TYPE == "SGD":
        G_optim = optim.SGD(net.parameters(),
                            lr=cfg.OPTIMIZER.LEARNING_RATE_G,
                            momentum=0.98, weight_decay=0.0001, nesterov=True)
    else:
        raise ValueError("Optimizer type not supported")
    # load pretrained model
    print('Start loading pretrained model')
    checkpoint = torch.load(cfg.TRAIN.PRETRAINPATH, map_location=torch.device('cpu'))
    print('*** using preTrain model: %s ***' % cfg.TRAIN.PRETRAINPATH)
    
    pretrained_dict = checkpoint['model_state_dict']
    # Update parameters for G now
    model_dict = net.state_dict()
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    
    if cfg.MEAN_TEACHER.use_mt:
        print("Create ema-net..")  
        
        # init ema-network. This model is utilized to generate pseudo-label.
        ema_net = MinkUNet34(cfg.MODEL_G).to(device)
        
        # Update parameters for G old
        model_dict = ema_net.state_dict()  
        model_dict.update(pretrained_dict)
        ema_net.load_state_dict(model_dict)
        print('finish update pretrained parameters')
    else:
        ema_net = None
        
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    
    trainer_file = importlib.import_module('trainers.' + cfg.TRAIN.STAGE)
    trainer = trainer_file.Trainer(cfg,
                                    net, ema_net,
                                    G_optim,
                                    logger, tf_writer, device)
    
    trainer.train()


if __name__ == '__main__':
    main()
  