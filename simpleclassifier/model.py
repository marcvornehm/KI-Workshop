from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn

from .data import SimpleDataset


Relu = "activation.relu"
Sigmoid = "activation.sigmoid"
Tanh = "activation.tanh"
NoActivation = "activation.none"


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation: str):
        super(ConvolutionalLayer, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        global Relu, Sigmoid, Tanh
        if activation == Relu:
            self.act = nn.ReLU()
        elif activation == Sigmoid:
            self.act = nn.Sigmoid()
        elif activation == Tanh:
            self.act = nn.Tanh()
        elif activation == NoActivation:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Invalid activation function: {activation}")
        # self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        # x = self.batchnorm(x)
        return x


class PoolingLayer(nn.Module):
    def __init__(self, kernel_size: int):
        super(PoolingLayer, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x):
        x = self.pool(x)
        return x


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: str):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        global Relu, Sigmoid, Tanh
        if activation == Relu:
            self.act = nn.ReLU()
        elif activation == Sigmoid:
            self.act = nn.Sigmoid()
        elif activation == Tanh:
            self.act = nn.Tanh()
        elif activation == NoActivation:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Invalid activation function: {activation}")

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x


def compute_flattened_features(conv_config: Sequence[nn.Module]):
    SimpleDataset.image_size
    x = torch.zeros((1, 1, 256, 256))
    for layer in conv_config:
        x = layer(x)
    return x.numel()


def create_network(conv_config: Sequence[nn.Module], fc_config: Sequence[nn.Module]):
    layers = OrderedDict()
    for i, layer in enumerate(conv_config):
        layers[f"layer_{i}"] = layer
    layers["flatten"] = nn.Flatten()
    for i, layer in enumerate(fc_config):
        layers[f"layer_{i+len(conv_config)}"] = layer
    return nn.Sequential(layers)
