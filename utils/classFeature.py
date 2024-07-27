import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


      
class Class_Features:
    def __init__(self, cfg):
        self.num_classes = cfg.MODEL_G.NUM_CLASSES

        self.class_features = [[] for i in range(self.num_classes)]
        self.num = np.zeros(self.num_classes)

        # copy from adaptation_modelv2.py
        self.Proto = torch.zeros([self.num_classes, 96])
        self.Amount = torch.zeros([self.num_classes])
        self.proto_momentum = cfg.PROTOTYPE.PROTOTYPE_EMA_STEPLR
        self.smoothloss = nn.SmoothL1Loss()
        # self.smoothloss = nn.MSELoss()# nn.MSELoss()
      
    def update_objective_SingleVector(self, id, vector,
                                        name='moving_average',
                                        start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.Amount[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.Proto[id] = self.Proto[id] * self.proto_momentum + (
                        1 - self.proto_momentum) * vector.squeeze()
            self.Amount[id] += 1
            self.Amount[id] = min(self.Amount[id], 3000)
        elif name == 'mean':
            # 以均值的方式得到原型，第一步 目前的特征原型 * 目前总数量 + 新来的特征
            self.Proto[id] = self.Proto[id] * self.Amount[id] + vector.squeeze()
            self.Amount[id] += 1  # 目前总数量 + 1
            # 加上新特征的目前特征原型 / 已改变的目前总数量
            self.Proto[id] = self.Proto[id] / self.Amount[id]
            self.Amount[id] = min(self.Amount[id], 3000)  # 最大目标总数量设置为3000
            pass
        else:
            raise NotImplementedError(
                'no such updating way of objective vectors {}'.format(name))

    def process_label(self, label):
        # label = label.unsqueeze(1) torch.Size([1, 160000, 1])
        label = label.permute(0, 2, 1).contiguous()
        batch, C, n = label.size()
        pred1 = torch.zeros(batch, self.num_classes, n).cuda()
        id = torch.where(label < self.num_classes, label,
                         torch.Tensor([0]).cuda())
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def calculate_mean_vector(self,
                              feat_cls_,
                              outputs,
                              labels_val=None,
                              min_dist_inds=None,
                              src_weight_map=None):
        
        feat_cls = feat_cls_.F.permute(1, 0).unsqueeze(0).contiguous()
        outputs = outputs.F.unsqueeze(0)

        if labels_val is not None:
            labels_val = labels_val.unsqueeze(0).unsqueeze(-1)
        
        # feat_cls = feat_cls.squeeze(0)
        outputs_softmax = F.softmax(outputs, dim=2)
        outputs_argmax = outputs_softmax.argmax(dim=2, keepdim=True)

        if labels_val == None:
            thresh = self.cfg.PROTOTYPE.PSELAB_THRESH
            conf = outputs_softmax.max(dim=2, keepdim=True)[0]
            mask = conf.ge(thresh)
            outputs_argmax = outputs_argmax * mask

            if min_dist_inds is not None and \
                self.cfg.PROTOTYPE.TGT_GENPSEUDO_USEMASK:
                dis_mask = min_dist_inds == outputs_argmax.view(-1)
                outputs_argmax = outputs_argmax * dis_mask.view_as(outputs_argmax)

            outputs_pred = self.process_label(outputs_argmax.float())
        else:
            labels_expanded = self.process_label(labels_val.float())
            outputs_argmax = self.process_label(outputs_argmax.float())
            outputs_pred = outputs_argmax * labels_expanded

        vectors = []
        ids = []  
        scale_factor = F.adaptive_avg_pool1d(outputs_pred, 1)
        """
        check
        (s.sum(1) / outputs_pred[:, t, :].sum()).view(-1)
        (F.adaptive_avg_pool1d(s, 1) / scale_factor[n][t]).view(-1)
        """
        # V1 cal feat
        for n in range(feat_cls.size()[0]):
            for t in range(self.num_classes):
                if t == 0:
                    continue
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                if src_weight_map == None:
                    s = feat_cls[n] * outputs_pred[n][t]  # # * mask[n]
                    s = F.adaptive_avg_pool1d(s, 1) / scale_factor[n][t]
                else:
                    s = feat_cls[n] * (1 + src_weight_map.view(-1)) * outputs_pred[n][t]
                    s = (s.sum(1) / ((1 + src_weight_map.view(-1)) * outputs_pred[n][t]).sum()).view(-1, 1)
                # s = F.adaptive_avg_pool1d(s, 1) / scale_factor[n][t]  # 特征均值 / t类在一张图像上的占比
                # s = s / (src_weight_map *  outputs_pred[n][t])
                vectors.append(s)
                ids.append(t)
               
        return vectors, ids

    def class_vectors_LDA(self, ids, vectors):
        """ 每个样本得到一个prototype 然后再计算LDA loss """
        loss = torch.Tensor([0]).cuda()
        for i in range(len(ids)):
            if ids[i] in [0]:
                continue
            intra_loss = self.smoothloss(vectors[i].squeeze(), self.Proto[ids[i]])
            # other_centers_mean = torch.cat([self.Proto[1:i], self.Proto[i + 1:]], dim=0).mean(0)
            # inter_loss = self.smoothloss(vectors[i].squeeze(), other_centers_mean)
            # class_i_loss = intra_loss / inter_loss 

            loss = loss + intra_loss
            
        loss = loss / len(ids)  

        return loss

    def class_points_LDA(self, ft, lab):
        """ 逐点计算LDA损失 """
        loss = torch.Tensor([0]).cuda()
        unique_lab = torch.unique(lab)
        valid_lab_count = 0
        intra_loss = 0

        for i in unique_lab:
            if i == 0:
                continue
            if (lab == i).sum() < 30:
                continue

            valid_lab_count += 1
            temp_ft = ft.F[lab == i, :]
            # V3
            intra_loss = intra_loss + self.smoothloss(temp_ft, self.Proto[i].expand_as(temp_ft))
          
        loss = intra_loss
        loss = loss / valid_lab_count

        return loss

    def tgt_class_points_LDA(self, ft, lab):
        """ target doamin逐点计算LDA目标域损失
        专门写一个函数的原因:考虑到之后可能要reweight这个loss
        所以没有直接取均值
        """
        intra_loss = torch.zeros(len(lab)).cuda()
        unique_lab = torch.unique(lab)
        valid_lab_count = 0
   
        for i in unique_lab:
            if i == 0:
                continue
            if (lab == i).sum() < 30:
                continue
            # V1
            valid_lab_count += 1
            temp_ft = ft.F[lab == i, :]
            intra_loss[lab == i] = F.smooth_l1_loss(temp_ft, self.Proto[i].expand_as(temp_ft), reduction='none').mean(1)
           
        loss = intra_loss
        loss = loss / valid_lab_count

        return loss


    def class_vectors_alignment(self, ids, vectors):
        loss = torch.Tensor([0]).cuda()
        for i in range(len(ids)):
            if ids[i] in [0]:
                continue
            # new_loss = self.smoothloss(vectors[i].squeeze().cuda(), torch.Tensor(self.src_centers.Proto[ids[i]]).cuda())
            new_loss = self.smoothloss(vectors[i].squeeze(), self.Proto[ids[i]])
          
            loss = loss + new_loss
        loss = loss / len(ids)  # * 10
        pass
        return loss

    def class_MMD_alignment(self, ids, vectors):
        sigma_list = [0.01, 0.1, 1, 10, 100]

        loss = torch.Tensor([0]).cuda()
        for i in range(len(ids)):
            if ids[i] in [0]:
                continue
            temp_loss = mix_rbf_mmd2(vectors[i].unsqueeze(0), self.Proto[ids[i]].unsqueeze(0), sigma_list)
            loss = loss + temp_loss

        loss = loss / len(ids)  # * 10
        pass
        return loss


class prototype_dist_estimator():
    def __init__(self, cfg, feature_num, resume_path=None):
        super(prototype_dist_estimator, self).__init__()

        self.cfg = cfg
        self.IGNORE_LABEL = 0
        self.class_num = cfg.MODEL_G.NUM_CLASSES
        self.feature_num = feature_num
        
        # momentum 
        self.use_momentum = True if cfg.PROTOTYPE.SRCPROTOTYPE_UPDATE_MODE == 'moving_average' else False
        self.momentum = cfg.PROTOTYPE.PROTOTYPE_EMA_STEPLR

        # init prototype
        self.init(feature_num=self.feature_num, resume=resume_path)

    def init(self, feature_num, resume):
        if resume:
            if feature_num == self.cfg.MODEL_G.NUM_CLASSES:
                resume = os.path.join(resume, 'prototype_out_dist.pth')
            elif feature_num == self.feature_num:
                resume = os.path.join(resume, 'prototype_feat_dist.pth')
            else:
                raise RuntimeError("Feature_num not available: {}".format(feature_num))
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            self.Proto = checkpoint['Proto'].cuda(non_blocking=True)
            self.Amount = checkpoint['Amount'].cuda(non_blocking=True)
        else:
            self.Proto = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
            self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)

    def update(self, features, labels=None):
        features = features
       
        if labels == None: # target domain without labels
            # pred = pred.F # .argmax(1)
            pred_argmax = pred.argmax(dim=1)

            # pred_softmax = F.softmax(pred, dim=1)

            # thresh = self.cfg.PROTOTYPE.PSELAB_THRESH
            # conf = pred_softmax.max(dim=1)[0]
            # mask = conf.ge(thresh)

            # labels = pred_argmax * mask
            labels = pred_argmax
        # else:
            # mask = pred_argmax == labels
        mask = labels != self.IGNORE_LABEL

        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        features = features[mask]
        if not self.use_momentum:
            N, A = features.size()
            C = self.class_num
            # refer to SDCA for fast implementation
            features = features.view(N, 1, A).expand(N, C, A)
            onehot = torch.zeros(N, C).cuda()
            onehot.scatter_(1, labels.view(-1, 1), 1)
            NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
            features_by_sort = features.mul(NxCxA_onehot)
            Amount_CXA = NxCxA_onehot.sum(0)
            Amount_CXA[Amount_CXA == 0] = 1
            mean = features_by_sort.sum(0) / Amount_CXA
            sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
            weight = sum_weight.div(
                sum_weight + self.Amount.view(C, 1).expand(C, A)
            )
            weight[sum_weight == 0] = 0
            self.Proto = (self.Proto.mul(1 - weight) + mean.mul(weight)).detach()
            self.Amount = self.Amount + onehot.sum(0)
            # 限制最多100000 太大了 更新很慢
            self.Amount[self.Amount > 100000] = 100000
            # self.Amount[i] = min(100000, self.Amount[i] + num_clsi)
        else:
            # momentum implementation
            ids_unique = labels.unique()
            for i in ids_unique:
                i = i.item()
                mask_i = (labels == i)
                feature = features[mask_i]
                feature = torch.mean(feature, dim=0)
                self.Amount[i] += len(mask_i)
                self.Proto[i, :] = (1 - self.momentum) * feature + self.Proto[i, :] * self.momentum 
        
    def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   name)




def class_MMD_alignment(self, ids, vectors):
    sigma_list = [0.01, 0.1, 1, 10, 100]

    loss = torch.Tensor([0]).cuda()
    for i in range(len(ids)):
        if ids[i] in [0]:
            continue
        temp_loss = mix_rbf_mmd2(vectors[i].unsqueeze(0), self.Proto[ids[i]].unsqueeze(0), sigma_list)
        loss = loss + temp_loss

    loss = loss / len(ids)  # * 10
    pass
    return loss
