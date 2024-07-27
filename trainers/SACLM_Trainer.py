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

from domain_mix.laserMix import LaserMix

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

        print("This is SAC LM Trainer")
        self.net_G = net_G
        
        if cfg.MEAN_TEACHER.use_mt:
            self.ema_G = ema_G
            self.create_ema_model(self.ema_G, self.net_G)
     
        self.G_optim = G_optim
        
        """ Define Loss Function """
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # seg loss
        
        self.laserMix = LaserMix(self.cfg)

        print("Init Trainer Done!")
        
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

            # 1. use teacher model to generate pseudo label
            self.tgt_BData['pseudo_label'] = self.get_pseudo_label()
            # mask data
            self.masked_batch = self.laserMix.mix(self.src_BData, self.tgt_BData)

            # update G
            src_loss = self.train_source()
            tgt_loss = self.train_target()
            all_loss = src_loss + tgt_loss
            all_loss.backward()
            
            self.G_optim.step()

            if self.cfg.MEAN_TEACHER.use_mt and \
                self.cfg.MEAN_TEACHER.alpha_ema > 0 and \
                    self.c_iter % self.cfg.MEAN_TEACHER.update_every == 0:
                self.update_ema_variables(self.ema_G, self.net_G)

            if self.c_iter % self.cfg.TRAIN.LOG_PERIOD == 0:
                print('iter:{0:6d}, '
                      'seg_Ls:{1:.4f}, '
                      'pse_seg_loss:{2:.4f}, '
                      'itr:{3:.3f}, '
                      'Exp:{4}'.format(self.c_iter,
                                       self.wb_dict['netG/seg_Loss'],
                                       self.wb_dict['netG/pse_seg_loss'],
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

    def train_source(self):# ===========train G ================
        if self.cfg.DATASET_SOURCE.use_aug_for_laserMix:
            src_labels = self.src_BData["aug_labels_mink"].cuda()
            src_G_in = ME.SparseTensor(coordinates=self.src_BData["aug_coords_mink"].int(),
                                          features=self.src_BData["aug_feats_mink"])
        else:
            src_labels = self.src_BData["labels_mink"].cuda()
            src_G_in = ME.SparseTensor(coordinates=self.src_BData["coords_mink"].int(),
                                          features=self.src_BData["feats_mink"])
                
        # Train with Source. compute source seg loss        
        src_outDict = self.net_G(src_G_in)

        all_src_loss = 0.
      
        # loss 1. main classifier CE loss
        src_seg_loss = self.criterion(src_outDict['sp_out'].F, src_labels)
        all_src_loss = all_src_loss + src_seg_loss
        
        self.wb_dict['netG/seg_Loss'] = src_seg_loss.mean()
        self.wb_dict['netG/all_src_loss'] = all_src_loss.mean()
        
        return all_src_loss
    
    def train_target(self):
        all_tgt_loss = 0.
        mix_stensor = ME.SparseTensor(coordinates=self.masked_batch["mixed_coors_1"].int(),
                                      features=self.masked_batch["mixed_feats_1"])
        mix_outDict = self.net_G(mix_stensor)
        mix_labels = self.masked_batch["mixed_lbls_1"].cuda()
        mix_loss = self.criterion(mix_outDict['sp_out'].F, mix_labels.long())
        all_tgt_loss = all_tgt_loss + mix_loss
        self.wb_dict['netG/pse_seg_loss'] = mix_loss.mean()

        # sac loss
        if self.cfg.TARGET_LOSS.lambda_sac > 0:
            with torch.no_grad():  # old-model generate pseudo-label
                tea_tgt_G_in = ME.SparseTensor(self.tgt_BData['feats_mink'],
                                               self.tgt_BData['coords_mink'])
                tea_raw_tgt_outDict = self.ema_G(tea_tgt_G_in)
                tea_raw_tgt_logit = tea_raw_tgt_outDict['sp_out']
                del_tgt_out = tea_raw_tgt_logit.F[self.tgt_BData['aug_del_mask']]
                raw2aug_tgt_out = del_tgt_out[self.tgt_BData['aug_unique_map']]
            
            # change raw tgt to aug
            tgt_G_in = ME.SparseTensor(self.tgt_BData['aug_feats_mink'], self.tgt_BData['aug_coords_mink'])
            stu_aug_tgt_outDict = self.net_G(tgt_G_in)
            stu_aug_tgt_logit = stu_aug_tgt_outDict['sp_out']
            sac_loss = F.kl_div(F.log_softmax(stu_aug_tgt_logit.F, dim=1),
                                F.softmax(raw2aug_tgt_out.detach(), dim=1))
            sac_loss = sac_loss * self.cfg.TARGET_LOSS.lambda_sac 
            all_tgt_loss = all_tgt_loss + sac_loss
            self.wb_dict['netG/sac_loss'] = sac_loss.mean()

        return all_tgt_loss

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

        self.ema_G.eval()
     
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
    
    