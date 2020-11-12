# -*- coding: utf-8 -*-
# 在pytorch的ResNet的基础上自定义的Resnet
import math

import torch
import torchvision
import torch.nn as nn


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, in_channels=3, depth=50, pre_trained=True, cut_after_avgpool=True, out_features=1024, final_pool=None):
        """

        :param in_channels 输入通道数
        :param depth: 网络深度
        :param pre_trained: 是否使用预训练的模型
        :param cut_after_avgpool: 是否在avgpool层之后截断网络
        :param out_features: 网络最后输出的特征向量的维度
        """
        super(ResNet, self).__init__()

        self.cut_after_avgpool = cut_after_avgpool
        self.final_pool = final_pool

        self.weight_default, self.bias_default = 1, 0.01

        if depth not in ResNet.__factory:
            raise KeyError('resnet网络深度设置错误！得到的值是{}，支持的有{}'.format(depth, ResNet.__factory.keys()))

        self.base_resnet = ResNet.__factory[depth](pretrained=pre_trained)

        out_planes = self.base_resnet.fc.in_features
        self.fc = nn.Linear(out_planes, out_features)            # 自定义fc层
        # self.fc_bn = nn.BatchNorm1d(out_features)                # 自定义bn层

        # 自定义层的初始化
        nn.init.kaiming_uniform_(self.fc.weight, mode="fan_out")
        nn.init.constant_(self.fc.bias, self.bias_default)
        # nn.init.constant_(self.fc_bn.weight, self.weight_default)
        # nn.init.constant_(self.fc_bn.bias, self.bias_default)

        if in_channels != 3:
            self.base_resnet.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7),
                                               stride=(2, 2), padding=(3, 3), bias=False)
            nn.init.kaiming_uniform_(self.base_resnet.conv1.weight, mode="fan_out")

        if not pre_trained:
            self.reset_params()

    def forward(self, x: torch.Tensor):
        """

        :param x: 输入
        :return:
        """
        n, c_in, h_in, w_in = x.shape
        # print(n, c_in, h_in, w_in)

        for name, module in self.base_resnet._modules.items():
            # print("name: " + name)
            # print("input: ", x.shape)
            x = module(x)
            # print("output: ", x.shape)
            # print("--------------------------------------------------------------------------------------------")

            if name == 'avgpool':
                break

        x = x.squeeze()                  # n x ? x 1 x 1 -> n x ?
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        # x = self.fc_bn(x)
        # print(x.shape)
        # x shape: (n, out_features)

        if self.final_pool is None:
            return x
        if self.final_pool == "avg":
            x = torch.mean(x, dim=0)
        elif self.final_pool == "max":
            x = torch.max(x, dim=0)[0]             # torch.max()不仅返回最大值，还返回最大值索引
        else:
            raise ValueError("不支持的最后特征池化类型: {}".format(self.final_pool))

        # x shape: (out_features)
        return x

    def reset_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, mode="fan_out")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, self.bias_default)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, self.weight_default)
                nn.init.constant_(layer.bias, self.bias_default)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.001)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, self.bias_default)

    def print_layers_info(self):
        print("-----------------------------model info started-----------------------------------")
        for name, module in self._modules.items():
            print(module)
        print("-----------------------------model info finished-----------------------------------")


if __name__ == '__main__':
    channel_num = 6
    input_x = torch.rand(size=(8, channel_num, 128, 64), dtype=torch.float)
    net = ResNet(in_channels=channel_num, depth=18, pre_trained=True)

    output_x = net(input_x)
    print(output_x.shape)
    net.print_layers_info()
