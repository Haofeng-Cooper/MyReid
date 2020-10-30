# -*- coding: utf-8 -*-
# 在pytorch的ResNet的基础上自定义的Resnet


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

    def __init__(self, depth=50, pre_trained=True, cut_after_avgpool=True, out_features=1024, final_pool=None):
        """

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
