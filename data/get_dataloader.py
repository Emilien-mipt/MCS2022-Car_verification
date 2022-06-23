import random

import numpy as np
import torch

from . import augmentations, dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(config):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """
    if config.train.full_training:
        print("Training on full data!")
        train_list = './datasets/CompCars/annotation/train_full.txt'
    else:
        train_list = './datasets/CompCars/annotation/train.txt'

    print("Preparing train reader...")
    train_dataset = dataset.CarsDataset(
        root=config.dataset.root,
        annotation_file=train_list,
        transforms=augmentations.get_train_aug(config),
        debug_mode=config.train.debug,
    )

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
    )

    val_dataset = dataset.CarsDataset(
        root=config.dataset.root,
        annotation_file=config.dataset.val_list,
        transforms=augmentations.get_val_aug(config),
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    print("Dataloaders are ready for training.")
    return train_loader, valid_loader
