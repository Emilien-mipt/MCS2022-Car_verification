import timm
import torch.nn as nn

from .head import AdaCos, ArcFace, CosFace


class MCSNet(nn.Module):
    def __init__(
        self,
        n_classes,
        model_name,
        use_fc=False,
        fc_dim=512,
        dropout=0.5,
        loss_module="softmax",
        s=64.0,
        margin=0.50,
        theta_zero=0.785,
        pretrained=True,
        device_id=None,
    ):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
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

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        if use_fc:
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            # Insert activation function?
            self.dropout = nn.Dropout(p=dropout)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == "arcface":
            self.final = ArcFace(
                in_features=final_in_features,
                out_features=n_classes,
                device_id=device_id,
                s=s,
                m=margin,
                easy_margin=False,
            )
        elif loss_module == "cosface":
            self.final = CosFace(
                in_features=final_in_features,
                out_features=n_classes,
                device_id=device_id,
                s=s,
                m=margin,
            )
        elif loss_module == "adacos":
            self.final = AdaCos(
                final_in_features, n_classes, m=margin, theta_zero=theta_zero
            )
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_features(x)
        if self.loss_module in ("arcface", "cosface", "adacos"):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.fc(x)
            x = self.bn(x)
            x = self.dropout(x)
        return x
