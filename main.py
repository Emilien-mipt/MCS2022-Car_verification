import argparse
import os
import os.path as osp
import random
import shutil
import sys

import numpy as np
import torch
import yaml
from pytorch_metric_learning import losses
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from data import get_dataloader
from models import models
from train import train, validation
from utils import convert_dict_to_tuple


def main(args: argparse.Namespace) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    outdir = osp.join(config.outdir, config.exp_name)

    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    shutil.copy2(args.cfg, outdir)

    tb = SummaryWriter(outdir)

    train_loader, val_loader = get_dataloader.get_dataloaders(config)

    print("Loading model...")
    net = models.load_timm_model(config)
    if config.num_gpu > 1:
        net = torch.nn.DataParallel(net)
    print("Done.")

    criterion, optimizer, scheduler = utils.get_training_parameters(config, net)
    train_epoch = tqdm(
        range(config.train.n_epoch), dynamic_ncols=True, desc="Epochs", position=0
    )

    ### pytorch-metric-learning stuff ###
    loss_func = losses.SubCenterArcFaceLoss(
        num_classes=config.dataset.num_of_classes,
        embedding_size=config.model.embedding_size,
    ).to("cuda")
    _, loss_optimizer, loss_scheduler = utils.get_training_parameters(config, loss_func)
    ### pytorch-metric-learning stuff ###

    # main process
    best_acc = 0.0
    for epoch in train_epoch:
        avg_train_loss, avg_train_acc = train(
            net, train_loader, loss_func, optimizer, loss_optimizer, config, epoch
        )
        epoch_avg_loss, epoch_avg_acc = validation(net, val_loader, loss_func, epoch)

        cur_lr = optimizer.param_groups[0]["lr"]
        tb.add_scalar("Learning rate", cur_lr, epoch + 1)
        tb.add_scalar("Train Loss", avg_train_loss, epoch + 1)
        tb.add_scalar("Train accuracy", avg_train_acc, epoch + 1)
        tb.add_scalar("Val Loss", epoch_avg_loss, epoch + 1)
        tb.add_scalar("Val accuracy score", epoch_avg_acc, epoch + 1)

        if epoch_avg_acc >= best_acc:
            utils.save_checkpoint(net, epoch, outdir)
            best_acc = epoch_avg_acc
        scheduler.step()
        loss_scheduler.step()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="Path to config file.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
