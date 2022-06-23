import argparse
import os
import os.path as osp
import shutil
import sys
import warnings

import torch
import yaml
from pytorch_metric_learning import losses
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from data import get_dataloader
from models.head import CircleLoss, InstanceLoss
from models.model import MCSNet
from train import train, validation
from utils import (add_weight_decay, convert_dict_to_tuple, get_optimizer,
                   get_scheduler, set_seed)


def main(args: argparse.Namespace) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)
    device_id = config.gpu_id
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("Selected device: ", device)

    seed = config.dataset.seed
    set_seed(seed)

    n_classes = config.dataset.num_of_classes

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
    model_params = {
        "model_name": config.model.model_name,
        "pretrained": config.model.pretrained,
        "fc_dim": config.model.fc_dim,
    }
    print("Model params: ", model_params)
    net = MCSNet(n_classes=n_classes, **model_params)
    net.to(device)

    embedding_size = net.fc_layer.linear

    print(f"Embedding size: {embedding_size}")

    # Dictionary with losses for metric learning. Set parameters here:
    loss_dict = {
        "arcface": losses.ArcFaceLoss(
            num_classes=n_classes, embedding_size=embedding_size
        ),
        "cosface": losses.CosFaceLoss(
            num_classes=n_classes, embedding_size=embedding_size
        ),
        "circle": CircleLoss(m=0.25, gamma=32),
        "lifted": losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0),
        "contrast": losses.ContrastiveLoss(pos_margin=0, neg_margin=1),
        "instance": InstanceLoss(gamma=32),
        "sphere": losses.SphereFaceLoss(
            num_classes=n_classes, embedding_size=embedding_size, margin=4
        ),
    }

    selected_losses = {
        loss: loss_dict[loss] for loss in loss_dict if loss in config.model.loss_modules
    }
    print("Selected losses:", selected_losses.keys())

    criterion = torch.nn.CrossEntropyLoss().to("cuda")

    no_decay_parameters, decay_parameters = add_weight_decay(net)
    optimizer = get_optimizer(
        config,
        decay_parameters=decay_parameters,
        no_decay_parameters=no_decay_parameters,
    )

    scheduler = get_scheduler(config, optimizer)

    train_epoch = tqdm(
        range(config.train.n_epoch), dynamic_ncols=True, desc="Epochs", position=0
    )

    # main process
    best_acc = 0.0
    for epoch in train_epoch:
        avg_train_loss, avg_train_acc, avg_train_top5 = train(
            model=net,
            selected_losses=selected_losses,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            device=device,
            scheduler=None,
        )
        tb.add_scalar("Train Loss", avg_train_loss, epoch + 1)
        tb.add_scalar("Train accuracy", avg_train_acc, epoch + 1)
        tb.add_scalar("Train Prec@5", avg_train_top5, epoch + 1)

        if (
            config.train.full_training or config.train.debug
        ):  # If training on full data or debug mode, do not validate
            utils.save_checkpoint(net, config.model.model_name, epoch, outdir)
        else:
            epoch_avg_loss, epoch_avg_acc, avg_val_top5 = validation(
                model=net,
                val_loader=val_loader,
                criterion=criterion,
                epoch=epoch,
                device=device,
            )
            tb.add_scalar("Val Loss", epoch_avg_loss, epoch + 1)
            tb.add_scalar("Val accuracy score", epoch_avg_acc, epoch + 1)
            tb.add_scalar("Val Prec@5", avg_val_top5, epoch + 1)

            if epoch_avg_acc >= best_acc:
                utils.save_checkpoint(net, config.model.model_name, epoch, outdir)
                best_acc = epoch_avg_acc

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {cur_lr}")
        tb.add_scalar("Learning rate", cur_lr, epoch + 1)

        scheduler.step()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="Path to config file.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
