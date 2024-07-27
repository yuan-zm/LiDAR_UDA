import torch
import torch.nn.functional as F
import torch.nn as nn

class ProtoContrast_loss(nn.Module):
    def __init__(self, cfg, class_weight=None):
        super(ProtoContrast_loss, self).__init__()
        self.cfg = cfg
        self.class_weight = class_weight

        self.TAU = cfg.CONTRAST.TAU
     
        self.balace_sel = cfg.CONTRAST.balace_sel
        self.max_pcs_per_class = cfg.CONTRAST.sel_pcs_per_class if cfg.CONTRAST.balace_sel else None

        # self.ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()) 
        # self.ce_criterion = nn.CrossEntropyLoss(ignore_index=-100) 

        self.num_classes = self.cfg.MODEL_G.NUM_CLASSES
        self.ignore_label = 0

    def forward(self, feat, proto, labels, predict):
        # predict = predict.detach()
        assert not proto.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        assert feat.dim() == 2
        assert labels.dim() == 1
        # remove IGNORE_LABEL pixels
        mask = (labels != self.ignore_label)
        labels = labels[mask]
        feat = feat[mask]
        
        if self.balace_sel:
            predict = predict[mask]
            max_pcs_per_class = self.max_pcs_per_class 
            # class balance select
            sample_idx = []
            uni_labs = torch.unique(labels)
            count = 0
            for cur_lab in uni_labs:
                if cur_lab == -100:
                    continue
                cur_sidx = torch.where(labels == cur_lab)[0]
                snum = cur_sidx.__len__()

                if snum < max_pcs_per_class:
                    sample_idx.append(cur_sidx)
                else:
                    hard_indices = ((labels == cur_lab) & (predict != cur_lab)).nonzero()
                    easy_indices = ((labels == cur_lab) & (predict == cur_lab)).nonzero()
                    num_hard = hard_indices.shape[0]
                    num_easy = easy_indices.shape[0]
                
                    sample_each_class = max_pcs_per_class
                    if num_hard >= sample_each_class // 2 and num_easy >= sample_each_class // 2:
                        num_hard_keep = sample_each_class // 2
                        num_easy_keep = sample_each_class - num_hard_keep
                    elif num_hard >= sample_each_class // 2:
                        num_easy_keep = num_easy
                        num_hard_keep = sample_each_class - num_easy_keep
                    elif num_easy >= sample_each_class // 2:
                        num_hard_keep = num_hard
                        num_easy_keep = sample_each_class - num_hard_keep

                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    cur_sidx = torch.cat((hard_indices, easy_indices), dim=0).view(-1)
                    
                    sample_idx.append(cur_sidx)

                count += len(cur_sidx)

            num_rest = max_pcs_per_class * len(uni_labs) - count
            cur_sidx = torch.randperm(feat.shape[0], device='cuda')[:num_rest]
            sample_idx.append(cur_sidx)

            sample_idx = torch.cat(sample_idx) 
            sel_fetas = feat[sample_idx]           
            sel_labs = labels[sample_idx]
        else:
            sel_fetas = feat      
            sel_labs = labels
        
        sel_fetas = F.normalize(sel_fetas, p=2, dim=1)
        Proto = F.normalize(proto, p=2, dim=1)

        logits = sel_fetas.mm(Proto.permute(1, 0).contiguous())
        logits = logits / self.TAU

        ce_criterion = nn.CrossEntropyLoss(ignore_index=0, weight=self.class_weight) # 
        loss = ce_criterion(logits, sel_labs)
        return loss
    
    

def huber_loss_DADA(pred, label, reg_threshold=0.1):
  
    pred = pred.F  # .view(1, -1)

    pred = pred.view(-1, 1)
    label = label.view(-1, 1)
    n, c = pred.size()
    assert c == 1

    pred = pred.squeeze()
    label = label.squeeze()

    adiff = torch.abs(pred - label)
    batch_max = reg_threshold * torch.max(adiff).item()

    # Computes \text{input} \leq \text{other}input≤other element-wise.
    # compute adiff < batch_max
    t2_mask = adiff.le(batch_max).float()

    # Computes \text{input} > \text{other}input>other element-wise.
    t1_mask = adiff.gt(batch_max).float()

    t1 = adiff * t1_mask

    t2 = (adiff * adiff + batch_max * batch_max) / (2 * batch_max)
    t2 = t2 * t2_mask

    return (torch.sum(t1) + torch.sum(t2)) / torch.numel(pred.data)



def cal_tgt_PPD_loss(cfg,
                      src_centers, 
                      tgt_logits_Tea, 
                      tgt_ft_Stu,
                      tf2sp_dis,
                      mask_ent=None,
                      weight_map=None):
    """
    1. 根据每个点目前teachermodel的伪标签我们可以的到属于哪个类别
    2. 若这个类别和distance保持一致 则计算
    3. 使用预测不确定性 对 每个点的loss 进行reweight
    """

    pselab_thresh = cfg.PROTOTYPE.PSELAB_THRESH
    tgt_GenPseudo_useMask = cfg.TGT_LOSS.TGT_PPD_MASK
    tgt_PPD_reweight = cfg.TGT_LOSS.TGT_PPD_REWEIGHT

    # 1.根据每个点目前teachermodel的伪标签我们可以的到属于哪个类别
    outputs_softmax = F.softmax(tgt_logits_Tea.F.unsqueeze(0).detach(), dim=2)
    outputs_argmax = outputs_softmax.argmax(dim=2, keepdim=True)
    conf = outputs_softmax.max(dim=2, keepdim=True)[0]
    mask = conf.ge(pselab_thresh)
    outputs_argmax = outputs_argmax * mask
    # 2. 若这个类别和distance保持一致 则计算
    if tgt_GenPseudo_useMask:
        # min_dist_inds = tf2sp_dis.min(1)[1]
        # mask = min_dist_inds == outputs_argmax.view(-1)
        outputs_argmax = outputs_argmax.view(-1) * mask_ent
        
    outputs_argmax = outputs_argmax.view(-1)

    PPD_loss = src_centers.tgt_class_points_LDA(tgt_ft_Stu, outputs_argmax)
    
    cossim_sum = 0
    # 是否使用weightmap来reweight
    if weight_map is not None and tgt_PPD_reweight:
        br_PPD_loss = PPD_loss.sum().detach() / (outputs_argmax != 0).sum()
        
        new_weightmap = 1 - weight_map.view(-1)
        if (new_weightmap < 0).sum() > 0:
            print('(1 - weight_map.view(-1)) < 0')
            print('((1 - weight_map.view(-1) < 0).sum()', (new_weightmap < 0).sum())
            new_weightmap = new_weightmap.clamp(min=0)

        PPD_loss = PPD_loss * new_weightmap
        PPD_loss = PPD_loss.sum() / (outputs_argmax != 0).sum()
    elif cfg.TGT_LOSS.TGT_PPD_REWEIGHT_COS:
        br_PPD_loss = PPD_loss.sum().detach() / (outputs_argmax != 0).sum()

        proto_ft = torch.gather(src_centers.Proto, dim=0, index=outputs_argmax.unsqueeze(1).expand(-1, src_centers.Proto.size()[1]))
        cossim = F.cosine_similarity(proto_ft, tgt_ft_Stu.F)
        cossim = 0.5 + 0.5 * cossim
        if cossim[cossim < 0].sum() != 0:
            print('cossim < 0 sum', (cossim < 0).sum())
        cossim[cossim < 0] = 0
        cossim_sum = cossim.sum()
        PPD_loss = cossim * PPD_loss
        PPD_loss = PPD_loss.sum() / (outputs_argmax != 0).sum()
    else:
        br_PPD_loss = PPD_loss.sum().detach() / (outputs_argmax != 0).sum()  
        PPD_loss = PPD_loss.sum() / (outputs_argmax != 0).sum()   

    PPD_loss = PPD_loss * cfg.TGT_LOSS.TGT_PPD_WEIGHT

    return PPD_loss, br_PPD_loss, cossim_sum

def fisher_loss(x, centers, labels, num_classes,
                inter_class="global",
                intra_loss_weight=1.0, inter_loss_weight=1.0):

    """Args:x: feature matrix with shape (n_pcs, feat_dim).
                labels: ground truth labels with shape (n_pcs).
                intra_loss_weight: float, default=1.0"""
        
    n_pcs = x.size(0)
   
    distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n_pcs, num_classes) + \
        torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(num_classes, n_pcs).T
    distmat.addmm_(x, centers.t(), beta=1, alpha=-2)
    intra_loss = (torch.gather(distmat, 1, labels.unsqueeze(1).repeat(1, num_classes))[:, 0])
    # between class distance
    if inter_class == "global":
        global_center = torch.mean(centers, 0)
        inter_loss = torch.pow(torch.norm(centers - global_center, p=2, dim=1), 2)  # .sum()
    else:
        raise ValueError("invalid value for inter_class argument, must be one of [global, sample]. ")

    loss = intra_loss_weight * intra_loss[labels != 0].sum() / (inter_loss_weight * inter_loss[1:].sum())

    disp = inter_loss[1:].detach() / intra_loss[labels != 0].mean().detach() 
    norm_disp = (disp - disp.min()) / (disp.max() - disp.min())

    return loss, norm_disp
  



def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    mse_loss = torch.nn.MSELoss(reduction='none')

    # pred = pred.squeeze(-1)
    B, C, N = pred.shape
    # loss = -soft_label.float() * F.log_softmax(pred, dim=1)

    loss = mse_loss(pred, soft_label)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return 30 * torch.mean(pixel_weights * torch.sum(loss, dim=1)) + torch.mean(torch.sum(loss, dim=1)) * 0.4


class w_bce(nn.Module):

    def __init__(self, size_average=True):
        super(w_bce, self).__init__()
        self.size_average = size_average

    def weighted(self, inputs, target, weight, alpha, beta):
        if not (target.size() == inputs.size()):
            raise ValueError(
                "Target size ({}) must be the same as inputs size ({})".format(target.size(), inputs.size()))
        max_val = (-inputs).clamp(min=0)
        loss = inputs - inputs * target + max_val + \
            ((-max_val).exp() + (-inputs - max_val).exp()).log()

        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, inputs, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(inputs, target, weight, alpha, beta)
        else:
            return self.weighted(inputs, target, None, alpha, beta)


class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, device, scale=5.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()


class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, device, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()
