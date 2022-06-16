from torch import nn
from torchvision import models

import timm


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    arch = config.model.arch
    num_classes = config.dataset.num_of_classes
    if arch.startswith('resnet'):
        model = models.__dict__[arch](pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception('model type is not supported:', arch)
    model.to('cuda')
    return model


def load_timm_model(config):
    """
    The function of loading a timm model by name from a configuration file
    :param config:
    :param pretrained:
    :return:
    """
    model_arch = config.model.arch
    num_classes = config.dataset.num_of_classes
    model = timm.create_model(model_arch, pretrained=True, num_classes=num_classes)
    model.to('cuda')
    return model
