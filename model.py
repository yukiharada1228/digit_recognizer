import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import torchsummary


class MNISTResNet18(ResNet):

    def __init__(self):

        # Based on ResNet18
        super(MNISTResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)


class MNISTResNet34(ResNet):

    def __init__(self):

        # Based on ResNet34
        super(MNISTResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)


if __name__ == '__main__':

    model = MNISTResNet34()
    torchsummary.summary(model, input_size=(1, 28, 28))
