import os, torch, random, time, wandb, yaml
import numpy as np


project_name = os.path.basename(os.getcwd())


def save_val_log(name, iou_list, train_dataset, tf_writer, logger, cur_iter):
        tb_name = 'valie_' + name + "/"
        s = name + ' IoU: \n'
        for ci, iou_tmp in enumerate(iou_list):
            s += '{:5.2f} '.format(100 * iou_tmp)
            class_name = train_dataset.label_name[ci]
            s += ' ' + class_name + ' '
            tf_writer.add_scalar(tb_name + class_name, 100 * iou_tmp, cur_iter)
        logger.info(s)


def save_best_check(net_G,
                    G_optim, src_centers,
                    cur_iter, logger, log_dir, name, iou):
    logger.info('**** Best mean {} val iou:{:.1f} ****'.format(name, iou * 100))
    filename = 'checkpoint_val_' + name + '.tar'
    fname = os.path.join(log_dir, filename)
    save_checkpoint(fname, net_G, 
                    G_optim, src_centers, cur_iter)


def save_checkpoint(fname, net_G, 
                    G_optim,
                    src_centers, cur_iter):
    save_dict = {
        'cur_iter': cur_iter + 1,  # after training one epoch, the start_epoch should be epoch+1
        'G_optim_state_dict': G_optim.state_dict(),
    }
  
    if src_centers is not None:
        save_dict['src_centers_Proto'] = src_centers.Proto
        save_dict['src_centers_Amount'] = src_centers.Amount
    # with nn.DataParallel() the net is added as a submodule of DataParallel
    try:
        save_dict['model_state_dict'] = net_G.module.state_dict()
    except AttributeError:
        save_dict['model_state_dict'] = net_G.state_dict()

    torch.save(save_dict, fname)

def loadCheckPoint(CHECKPOINT_PATH, net, classifier, D_out,
                    G_optim, head_optim, D_out_optim):

    checkpoint = torch.load(CHECKPOINT_PATH)

    c_iter = checkpoint['epoch']

    net.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    D_out.load_state_dict(checkpoint['D_out_model_state_dict'])

    G_optim.load_state_dict(checkpoint['G_optim_state_dict'])
    head_optim.load_state_dict(checkpoint['head_optim_state_dict'])
    D_out_optim.load_state_dict(checkpoint['D_out_optim_state_dict'])

    return net, classifier, D_out, G_optim, head_optim, D_out_optim, c_iter

def clean_summary(filesuammry):
    """
    remove keys from wandb.log()
    Args:
        filesuammry:

    Returns:

    """
    keys = [k for k in filesuammry.keys() if not k.startswith('_')]
    for k in keys:
        filesuammry.__delitem__(k)
    return filesuammry

def classProperty2dict(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value):
            pr[name] = value
    return pr


def make_reproducible(iscuda=True, seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if iscuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # set True will make data load faster
        #   but, it will influence reproducible
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

def mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=False)

def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES'])
        # torch.backends.cudnn.benchmark = True # speed-up cudnn
        # torch.backends.cudnn.fastest = True # even more speed-up?
        hint('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])

    else:
        hint('Launching on CPU')

    return cuda


def hint(msg):
    timestamp = f'{time.strftime("%m/%d %H:%M:%S", time.localtime(time.time()))}'
    print('\033[1m' + project_name + ' >> ' + timestamp + ' >> ' + '\033[0m' + msg)



SK2P_color_dic = {
    0: [0, 0, 0],
    1: [200, 40, 255],
    2: [245, 150, 100],
    3: [0, 60, 135],
    4: [0, 175, 0],
    5: [75, 0, 75],
    6: [150, 240, 255],
    7: [75, 0, 175],
    8: [0, 200, 255],
    9: [255, 150, 255],
    10: [50, 120, 255],
    11: [245, 230, 100],
    12: [255, 0, 255],
    13: [30, 30, 255] 
}


N2SK_color_dic = {
    0: [0, 0, 0],
    1: [245, 150, 100],
    2: [245, 230, 100],
    3: [150, 60, 30],
    4: [180, 30, 80],
    5: [255, 80, 100],
    6: [30, 30, 255],
    7: [255, 0, 255],
    8: [75, 0, 75],
    9: [80, 240, 150],
    10: [0, 175, 0],
}

N2SK_color_dic_mayavi = {

    0:    [245, 150, 100, 255],  # [0,0,0] for teaser1
    1:  [245, 230, 100, 255],
    2: [150, 60, 30, 255],
    3:   [180, 30, 80, 255],
    4:   [255, 0, 0, 255],
    5:   [30, 30, 255, 255],  # [255,0,0] for teaser1
    6:   [200, 40, 255, 255],
    7:   [90, 30, 150, 255],
    8:   [255, 0, 255, 255],
    9:   [255, 150, 255, 255],
    10:   [75, 0, 75, 255],
    11:   [75, 0, 175, 255],
    12:   [0, 200, 255, 255],  # 12
    13:   [50, 120, 255, 255],
    14:   [0, 175, 0, 255],  # 15
    15:   [0, 60, 135, 255],
    16:  [80, 240, 150, 255],
    17:  [150, 240, 255, 255],
    18:  [0, 0, 255, 255],
    19: [255, 255, 255, 255]  # no label
}

syn2sk_color_dic = {
    0 : [0, 0, 0],
    1: [245, 150, 100],
    2: [245, 230, 100],
    3: [150, 60, 30],
    4: [180, 30, 80],
    5: [255, 0, 0],
    6: [30, 30, 255],
    7: [200, 40, 255],
    8: [90, 30, 150],
    9: [255, 0, 255],
    10: [255, 150, 255],
    11: [75, 0, 75],
    12: [75, 0, 175],
    13: [0, 200, 255],
    14: [50, 120, 255],
    15: [0, 175, 0],
    16: [0, 60, 135],
    17: [80, 240, 150],
    18: [150, 240, 255],
    19: [75, 0, 75],
}
