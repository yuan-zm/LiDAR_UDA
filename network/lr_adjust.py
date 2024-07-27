
POWER = 0.9


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def lr_warmup(base_lr, iter, max_iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)

# def adjust_learning_rate(lr, optimizer, i_iter, max_iter, PREHEAT_STEPS):
#     if i_iter < PREHEAT_STEPS:
#         lr = lr_warmup(lr, i_iter, max_iter, PREHEAT_STEPS)
#     else:
#         lr = lr_poly(lr, i_iter, max_iter, POWER)
#     optimizer.param_groups[0]['lr'] = lr
#     if len(optimizer.param_groups) > 1:
#         optimizer.param_groups[1]['lr'] = lr * 10
    
#     return optimizer


def adjust_learning_rate(lr, i_iter, max_iter, PREHEAT_STEPS):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(lr, i_iter, max_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(lr, i_iter, max_iter, POWER)
   
    return lr

def adjust_learning_rate_D(lr_D, i_iter, max_iter, PREHEAT_STEPS):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(lr_D, i_iter, max_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(lr_D, i_iter, max_iter, POWER)
   
    return lr