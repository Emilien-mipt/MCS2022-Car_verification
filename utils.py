import os
import random
from collections import OrderedDict, namedtuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple("GenericDict", dictionary.keys())(**dictionary)


def save_checkpoint(model, model_name, epoch, outdir):
    """Saves checkpoint to disk"""
    filename = f"{model_name}_{epoch}.pth"
    directory = outdir
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    state = OrderedDict(
        [
            ("state_dict", weights),
            ("epoch", epoch),
        ]
    )
    torch.save(state, filename)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_optimizer(config, decay_parameters, no_decay_parameters):
    if config.train.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": decay_parameters,
                    "weight_decay": config.train.weight_decay,
                },
                {"params": no_decay_parameters},
            ],
            lr=config.train.learning_rate,
            momentum=config.train.momentum,
        )
    elif config.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [
                {
                    "params": decay_parameters,
                    "weight_decay": config.train.weight_decay,
                },
                {"params": no_decay_parameters},
            ],
            lr=config.train.learning_rate,
        )
    else:
        raise Exception("Unknown type of optimizer: {}".format(config.train.optimizer))
    return optimizer


def get_scheduler(config, optimizer):
    if config.train.lr_scheduler.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            eta_min=config.train.lr_scheduler.eta_min,
            T_max=config.train.lr_scheduler.T_max,
        )
    elif config.train.lr_scheduler.name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.lr_scheduler.step_size,
            gamma=config.train.lr_scheduler.gamma,
        )
    elif config.train.lr_scheduler.name == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.train.lr_scheduler.milestones,
            gamma=config.train.lr_scheduler.gamma,
        )
    else:
        raise Exception(
            "Unknown type of lr schedule: {}".format(config.train.lr_schedule)
        )
    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self):
        return self.val, self.avg


def get_max_bbox(bboxes):
    bbox_sizes = [x[2] * x[3] for x in bboxes]
    max_bbox_index = np.argmax(bbox_sizes)
    return bboxes[max_bbox_index]


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Do not decay parameters for biases and batch-norm layers
def add_weight_decay(model, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return no_decay, decay


# Adjust LR for each training batch during warm up
def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params["lr"] = batch * init_lr / num_batch_warm_up


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
