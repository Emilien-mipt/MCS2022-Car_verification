import timm
import torch.nn as nn
from torch.nn import init


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


class FCLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        droprate=0.5,
        relu=False,
        bnorm=True,
        linear=512,
    ):
        super(FCLayer, self).__init__()
        linear_block = []
        if linear > 0:
            linear_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            linear_block += [nn.BatchNorm1d(linear)]
        if relu:
            linear_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            linear_block += [nn.Dropout(p=droprate)]

        linear_block = nn.Sequential(*linear_block)
        linear_block.apply(weights_init_kaiming)
        self.fc_layer = linear_block
        self.linear = linear

    def forward(self, x):
        x = self.fc_layer(x)
        return x


class Classifier(nn.Module):
    def __init__(
        self,
        input_dim,
        class_num,
    ):
        super(Classifier, self).__init__()
        classifier_block = []
        classifier_block += [nn.Linear(input_dim, class_num)]
        classifier_block = nn.Sequential(*classifier_block)
        classifier_block.apply(weights_init_classifier)
        self.classifier = classifier_block

    def forward(self, x):
        x = self.classifier(x)
        return x


class MCSNet(nn.Module):
    def __init__(
        self,
        n_classes,
        model_name,
        fc_dim=512,
        dropout=0.5,
        relu=False,
        bnorm=True,
        pretrained=True,
    ):
        super(MCSNet, self).__init__()
        print("Building Model Backbone for {} model".format(model_name))
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        print("Backbone has been built!")

        if "resnet" in model_name:
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layer = FCLayer(
            input_dim=final_in_features,
            droprate=dropout,
            relu=relu,
            bnorm=bnorm,
            linear=fc_dim,
        )
        self.classifier = Classifier(input_dim=fc_dim, class_num=n_classes)

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)
        features = self.fc_layer(x)
        return features

    def forward(self, x):
        feature = self.extract_features(x)
        logits = self.classifier(feature)
        return logits


class MCSNetTransformers(nn.Module):
    def __init__(
        self,
        n_classes,
        model_name,
        fc_dim=512,
        dropout=0.5,
        relu=False,
        bnorm=True,
        pretrained=True,
    ):
        super(MCSNetTransformers, self).__init__()
        print("Building Model Backbone for {} model".format(model_name))
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        print("Backbone has been built!")

        final_in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.fc_layer = FCLayer(
            input_dim=final_in_features,
            droprate=dropout,
            relu=relu,
            bnorm=bnorm,
            linear=fc_dim,
        )
        self.classifier = Classifier(input_dim=fc_dim, class_num=n_classes)

    def extract_features(self, x):
        x = self.backbone.forward_features(x)
        features = self.fc_layer(x)
        return features

    def forward(self, x):
        feature = self.extract_features(x)
        logits = self.classifier(feature)
        return logits
