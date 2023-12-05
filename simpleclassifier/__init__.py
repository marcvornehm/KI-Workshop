from .data import CardiacViewDataset, BrainTumorDataset, CatsVsDogsDataset
from .model import ConvolutionalLayer, PoolingLayer, FullyConnectedLayer, Relu, Sigmoid, Tanh, NoActivation, create_network, compute_flattened_features
from .trainer import train, test

import torch
torch.manual_seed(42)
