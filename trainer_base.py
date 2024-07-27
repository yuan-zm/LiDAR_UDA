# Common
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from dataset.get_dataloader import get_TV_dl
from utils import common as com
from validate_train import validater

from dataset.source_dataset import SourceDataset
from dataset.target_dataset import TargetDataset

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class Base_Trainer:
    def __init__(self, cfg, logger, tf_writer, device):

        self.cfg = cfg
        
        self.start_iter = 0
        self.ml_info = {'bt_tgt_spIoU': 0}
        self.cfg = cfg
        self.logger = logger
        self.tf_writer = tf_writer
        self.device = device
        
        """ Define Loss Function """
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # seg loss
     
        """  get_dataset & dataloader """
        self.init_dataloader()
        self.t_val_iter = self.cfg.TRAIN.T_VAL_ITER
        self.s_val_iter = self.cfg.TRAIN.S_VAL_ITER

        """ Other training parameters"""
        self.c_iter = 0  # Current Iter
        self.round = 0 # current round
        self.best_IoU_iter = 0
        self.best_IoU_after_saveIter = 0

    def update_ema_variables(self, ema_net, net):
        alpha_teacher = min(1 - 1 / (self.c_iter + 1), self.cfg.MEAN_TEACHER.alpha_ema)
        self.cur_alpha_teacher = alpha_teacher
        for ema_param, param in zip(ema_net.parameters(), net.parameters()):
            ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)
        for t, s in zip(ema_net.buffers(), net.buffers()):
            if not t.dtype == torch.int64:
                t.data.mul_(alpha_teacher).add_(s.data, alpha=1 - alpha_teacher)

    def create_ema_model(self, ema, net):
        print('create_ema_model G to current iter {}'.format(self.c_iter))
        for param_q, param_k in zip(net.parameters(), ema.parameters()):
            param_k.data = param_q.data.clone()
        for buffer_q, buffer_k in zip(net.buffers(), ema.buffers()):
            buffer_k.data = buffer_q.data.clone()
        ema.eval()
        for param in ema.parameters():
            param.requires_grad_(False)
        for param in ema.parameters():
            param.detach_()

    @staticmethod
    def send_data2GPU(batch_data):
        for key in batch_data:  # send data to gpu
            batch_data[key] = batch_data[key].cuda(non_blocking=True)
        return batch_data

    def init_dataloader(self):

        # init source dataloader
        src_tra_dset = SourceDataset(self.cfg, 'train')
        src_val_dset = SourceDataset(self.cfg, 'validation')
        self.src_TraDL, _ = get_TV_dl(self.cfg, src_tra_dset, src_val_dset)

        # init target dataloader
        tgt_tra_dset = TargetDataset(self.cfg, 'train')
        tgt_val_dset = TargetDataset(self.cfg, 'validation') 
        self.tgt_train_loader, _ = get_TV_dl(self.cfg, tgt_tra_dset, tgt_val_dset, domain='target')

        # init validater
        self.src_valer = validater(self.cfg, 'source', self.criterion, self.tf_writer, self.logger)
        self.tgt_valer = validater(self.cfg, 'target', self.criterion, self.tf_writer, self.logger)
        
    def get_pseudo_label(self):
        # 1. use teacher model to generate pseudo label
        with torch.no_grad():  # old-model generate pseudo-label
            tgt_G_in = ME.SparseTensor(self.tgt_BData['feats_mink'], self.tgt_BData['coords_mink'])
            tgt_emaG_outDict = self.ema_G(tgt_G_in)

        # 2. filter pseudo label with confidence 
        target_confidence_th = self.cfg.PSEUDO_LABEL.threshold # self.target_confidence_th
        target_pseudo = tgt_emaG_outDict['sp_out'].F
        target_pseudo = F.softmax(target_pseudo, dim=-1)
        target_conf, target_pseudo = target_pseudo.max(dim=-1)
        filtered_target_pseudo = torch.zeros_like(target_pseudo)
        valid_idx = target_conf > target_confidence_th
        filtered_target_pseudo[valid_idx] = target_pseudo[valid_idx]
        target_pseudo = filtered_target_pseudo.long()
        
        return target_pseudo