import torch.nn as nn
import torch.nn.functional as F

from deeplabv3.aspp import ASPP
from deeplabv3.resnet import ResNet18_OS8


class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()

        self.num_classes = 20

        self.resnet = ResNet18_OS8()  # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)

        output = self.aspp(feature_map)  # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))

        return output
