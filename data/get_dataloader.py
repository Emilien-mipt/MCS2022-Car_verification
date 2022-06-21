import torch

from . import augmentations, dataset


def get_dataloaders(config):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """
    print("Preparing train reader...")
    train_dataset = dataset.CarsDataset(
        root=config.dataset.root,
        annotation_file=config.dataset.train_list,
        transforms=augmentations.get_train_aug(config),
        debug_mode=config.train.debug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
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
