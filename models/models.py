import timm
import torch
from torch import nn
from torch.nn import init
from torchvision import models


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    arch = config.model.arch
    num_classes = config.dataset.num_of_classes
    if arch.startswith("resnet"):
        model = models.__dict__[arch](pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("model type is not supported:", arch)
    model.to("cuda")
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
    model.fc = torch.nn.Identity()
    model.to("cuda")
    return model


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.kaiming_normal_(
            m.weight.data, a=0, mode="fan_in"
        )  # For old pytorch, you may use kaiming_normal.
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
    elif classname.find("BatchNorm1d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, "bias") and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def activate_drop(m):
    classname = m.__class__.__name__
    if classname.find("Drop") != -1:
        m.p = 0.1
        m.inplace = True


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        class_num,
        droprate,
        relu=True,
        bnorm=True,
        linear=512,
        return_f=False,
    ):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


# Define the HRNet18-based Model
class MCSNet(nn.Module):
    def __init__(
        self, model_name, class_num, droprate=0.5, circle=False, linear_num=512
    ):
        super().__init__()
        model_ft = timm.create_model(model_name=model_name, pretrained=True)
        if "resnet" in model_name:
            final_in_features = model_ft.fc.in_features
            model_ft.fc = nn.Identity()
        else:
            final_in_features = model_ft.classifier.in_features
            model_ft.classifier = nn.Identity()
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(
            final_in_features,
            class_num,
            droprate,
            relu=True,
            bnorm=True,
            linear=linear_num,
            return_f=circle,
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
