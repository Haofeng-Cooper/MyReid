# -*- coding: utf-8 -*-
# 输出 resnet 各层细节信息和在既定输入下各层的输出size


import torch
from torchvision.models import resnet
from model.resnet import ResNet as MyResNet


def main():
    shape = (8, 3, 128, 64)                  # (b, c, h, w) PRID2011
    # shape = (8, 3, 256, 128)                 # Mars
    # model = resnet.resnet50(pretrained=True)
    model = resnet.resnet18(pretrained=True)

    x = torch.rand(size=shape, dtype=torch.float)
    for name, module in model._modules.items():
        print(name + ":")
        tout = module(x)
        print("input: ", x.shape, "\toutput: ", tout.shape)
        print(module)
        x = tout
        print("------------------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    shape = (8, 3, 128, 64)
    x = torch.rand(size=shape, dtype=torch.float)
    model = MyResNet()
    model.forward(x)
