import albumentations as A
import torchvision as tv
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


normalize = tv.transforms.Normalize(mean=MEAN, std=STD)


def get_train_aug(config):
    if config.dataset.augmentations == "default":
        train_augs = tv.transforms.Compose(
            [
                tv.transforms.RandomResizedCrop(config.dataset.input_size),
                # tv.transforms.CenterCrop(config.dataset.input_size),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    elif config.dataset.augmentations == "complex":
        train_augs = A.Compose(
            [
                A.RandomResizedCrop(
                    height=config.dataset.input_size, width=config.dataset.input_size
                ),
                A.HorizontalFlip(),
                A.OneOf(
                    [
                        A.GaussNoise,
                        A.ISONoise,
                        A.MultiplicativeNoise,
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.HueSaturationValue(10, 15, 10),
                        A.CLAHE(clip_limit=2),
                        A.RandomBrightnessContrast(),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.CoarseDropout(),
                        A.Cutout(),
                    ],
                    p=0.3,
                ),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ],
            p=1.0,
        )
    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return train_augs


def get_val_aug(config):
    if config.dataset.augmentations_valid == "default":
        val_augs = tv.transforms.Compose(
            [
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(config.dataset.input_size),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return val_augs
