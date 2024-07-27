# Common
import os
import wandb
import numpy as np
import time
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from network.lr_adjust import adjust_learning_rate
from utils import common as com

from domain_mix.cosMix import CoSMix
from trainer_base import Base_Trainer

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
class Trainer(Base_Trainer):
    def __init__(self,
                 cfg,
                 net_G, ema_G,
                 G_optim, 
                 logger, tf_writer, device):
        super().__init__(cfg, logger, tf_writer, device)

        print("This is a CoSMix trainer.")
        self.net_G = net_G
       
        self.G_optim = G_optim
    
        """ Define Loss Function """
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # seg loss
        if self.cfg.SOURCE_LOSS.lambda_lov > 0.:
            from network.lov_loss import Lovasz_loss
            self.lov_criterion = Lovasz_loss(ignore=0)
        
        if cfg.MEAN_TEACHER.use_mt:
            self.ema_G = ema_G
            self.create_ema_model(self.ema_G, self.net_G)
        
        self.cosMix = CoSMix(self.cfg)
        
        print("Init CoSMix Trainer Done.")
        
    def train(self):
        for epoch in range(self.cfg.TRAIN.MAX_EPOCHS):
            print("This is epoch: {}".format(epoch))

            self.train_one_epoch()
    
    def train_one_epoch(self):
        for tgt_BData in self.tgt_train_loader:          
            self.wb_dict = {}
            self.c_iter += 1
            start_t = time.time()
            self.set_lr()
            self.set_zero_grad()

            # send data to GPU
            src_BData = self.src_TraDL.next()
            self.src_BData = self.send_data2GPU(src_BData)
            self.tgt_BData = self.send_data2GPU(tgt_BData)

            """ tea-model forward the raw target scans """
            self.tgt_BData['pseudo_label'] = self.get_pseudo_label()
            
            # cosMix input scan
            masked_batch = self.cosMix.mix(self.src_BData, self.tgt_BData)

            all_loss = 0.
            # CoSMix do not train with raw source data 
            # 1. select source points to mixed with target scan
            s2t_stensor = ME.SparseTensor(coordinates=masked_batch["mixed_coors_tgt"].int(),
                                          features=masked_batch["mixed_feats_tgt"])
            s2t_outDict = self.net_G(s2t_stensor)
            s2t_logit = s2t_outDict['sp_out']
            s2t_labels = masked_batch["mixed_lbls_tgt"].cuda()
            s2t_loss = self.criterion(s2t_logit.F, s2t_labels.long())
            all_loss = all_loss + s2t_loss
            
            # 2. select target points to mixed with source scan
            t2s_stensor = ME.SparseTensor(coordinates=masked_batch["mixed_coors_src"].int(),
                                          features=masked_batch["mixed_feats_src"])
            t2s_outDict = self.net_G(t2s_stensor)
            t2s_logit = t2s_outDict['sp_out']
            t2s_labels = masked_batch["mixed_lbls_src"].cuda()
            t2s_loss = self.criterion(t2s_logit.F, t2s_labels.long())
            all_loss = all_loss + t2s_loss
            
            all_loss.backward()
            self.G_optim.step()
            
            self.wb_dict['netG/s2t_loss'] = s2t_loss.mean()
            self.wb_dict['netG/t2s_loss'] = t2s_loss.mean()

            if self.cfg.MEAN_TEACHER.use_mt and \
                self.cfg.MEAN_TEACHER.alpha_ema > 0 and \
                    self.c_iter % self.cfg.MEAN_TEACHER.update_every == 0:
                self.update_ema_variables(self.ema_G, self.net_G)
                
            if self.c_iter % self.cfg.TRAIN.LOG_PERIOD == 0:
                print('iter:{0:6d}, '
                      'src_Ls:{1:.4f}, '
                      'tgt_Ls:{2:.4f}, '
                      'itr:{3:.3f}, '
                      'Exp:{4}'.format(self.c_iter,
                                       self.wb_dict['netG/t2s_loss'],
                                       self.wb_dict['netG/s2t_loss'],
                                       time.time() - start_t,
                                       self.cfg.TRAIN.EXP_NAME))

                self.save_log()  # save logs

            if self.c_iter % self.t_val_iter == 0: # Traget domain val.
                self.valid_and_save()

            if self.c_iter % self.s_val_iter == 0: # Source domain val.
                _  = self.src_valer.rolling_predict(self.net_G, self.ema_G, self.c_iter, domain='src')

            if self.c_iter % 10 == 0:
                torch.cuda.empty_cache()

            if self.c_iter == self.cfg.TRAIN.MAX_ITERS:
                if self.c_iter % self.t_val_iter != 0:
                    self.valid_and_save()
                print("Finish training, this is max iter: {}".format(self.c_iter))
                quit()

        torch.cuda.empty_cache()

    def valid_and_save(self):
        cp_fn = os.path.join(self.cfg.TRAIN.MODEL_DIR, 'cp_current.tar')
        self.fast_save_CP(cp_fn)

        # If you want save model checkpoint, set cfg.TRAIN.SAVE_MORE_ITER = True
        if self.c_iter > self.cfg.TRAIN.SAVE_ITER and self.cfg.TRAIN.SAVE_MORE_ITER:
            cp_fn = os.path.join(self.cfg.TRAIN.MODEL_DIR, 'cp_{}_iter.tar'.format(self.c_iter))
            self.fast_save_CP(cp_fn)

        tgt_sp_iou = self.tgt_valer.rolling_predict(self.net_G, self.ema_G, self.c_iter, domain='tgt')

        if (tgt_sp_iou > self.best_IoU_after_saveIter and self.c_iter > self.cfg.TRAIN.SAVE_ITER) or \
                tgt_sp_iou > self.ml_info['bt_tgt_spIoU']:
            s_name = 'target_Sp'

            if (tgt_sp_iou > self.best_IoU_after_saveIter and self.c_iter > self.cfg.TRAIN.SAVE_ITER):
                self.best_IoU_after_saveIter = tgt_sp_iou
                s_name = 'target_Sp_After'

            self.best_IoU_iter = self.c_iter
            self.ml_info['bt_tgt_spIoU'] = tgt_sp_iou
            wandb.run.summary["bt_tgt_spIoU"] = tgt_sp_iou

            com.save_best_check(self.net_G, 
                                self.G_optim, None,
                                self.c_iter, self.logger,
                                self.cfg.TRAIN.MODEL_DIR, name=s_name,
                                iou=tgt_sp_iou)

        torch.cuda.empty_cache()
    
    def save_log(self):
        self.wb_dict['lr/lr_G'] = self.G_optim.state_dict()['param_groups'][0]['lr']

        for k, v in self.wb_dict.items():
            self.tf_writer.add_scalar(k, v, self.c_iter)
            wandb.log({k: v}, step=self.c_iter)

    def set_zero_grad(self):
        self.net_G.train()  # set model to training mode
        
        self.G_optim.zero_grad()

    def set_lr(self):
        current_lr_G = adjust_learning_rate(self.cfg.OPTIMIZER.LEARNING_RATE_G,
                                            self.c_iter, self.cfg.TRAIN.MAX_ITERS,
                                            self.cfg.TRAIN.PREHEAT_STEPS)

        for index in range(len(self.G_optim.param_groups)):
            self.G_optim.param_groups[index]['lr'] = current_lr_G
      
    def fast_save_CP(self, checkpoint_file):
        com.save_checkpoint(checkpoint_file,
                            self.net_G,
                            self.G_optim, 
                            None,
                            self.c_iter)
    
    
        