from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Dropout, Sequential

import numpy as np


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is:

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.

    If dropout is used, a dropout layer is added after every ReLU.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param: Dropout probability. Zero means no dropout.
        """
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.dropout = dropout                

        blocks = []
        Din = in_features
        for Dout in hidden_features:
            blocks.append(Linear(Din,Dout))
            
            blocks.append(ReLU())
            
            if self.dropout != 0:
                blocks.append(Dropout(self.dropout))
            
            Din = Dout
        
        blocks.append(Linear(Din,num_classes))

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        Cin = in_channels;
        assert len(self.filters) % self.pool_every == 0, 'length of filters must be a multiplier of pool_every'
        l = len(self.filters) // self.pool_every
        filters = [self.filters[x:x+self.pool_every] for x in np.arange(l)*self.pool_every]
        
        for lay in filters:
            for Cout in lay:
                layers += [nn.Conv2d(Cin,Cout,3,padding=1),nn.ReLU()]
                Cin = Cout
            layers += [nn.MaxPool2d(2)]
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        Din_ = self.filters[-1] * in_h * in_w
        pools = len(self.filters) / self.pool_every
        red = pow(4,pools)
        Din = int(Din_/red)

        for Dout in self.hidden_dims:
            layers += [nn.Linear(Din,Dout), nn.ReLU()]
            Din = Dout
        
        layers += [nn.Linear(Din,self.out_classes)]
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        x = self.feature_extractor(x)
        N = x.shape[0]
        x = x.view(N,-1)
        out = self.classifier(x)
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims, dynamic_dropout = True):
        self.dynamic_dropout = dynamic_dropout
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        Cin = in_channels;
        assert len(self.filters) % self.pool_every == 0, 'length of filters must be a multiplier of pool_every'
        l = len(self.filters) // self.pool_every
        filters = [self.filters[x:x+self.pool_every] for x in np.arange(l)*self.pool_every]
        

        dropout_p = 0.5
        to_add = 0 
        if self.dynamic_dropout:
            dropout_p = 0.2
            to_add = 0.5 / l 

        for lay in filters:
            for Cout in lay:
                layers.append(ResNetConvBlok(Cin, Cout))
                Cin = Cout
            layers.append(ResNetPullBlock(p=dropout_p))
            dropout_p += to_add

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        seq = nn.Sequential(*layers)
        return seq


    def _make_classifier(self):
        Din = self.filters[-1]
        layers = []
        for Dout in self.hidden_dims:
            layers += [nn.Linear(Din,Dout), nn.Softmax()]
            Din = Dout
        
        layers += [nn.Linear(Din,self.out_classes)]
        seq = nn.Sequential(*layers)
        return seq



class ResNetConvBlok(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut_path = nn.Sequential()
        # Check if spatial or channel dimentions changed along main path
        # If so, we need to adjust the dimensions of the shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = F.relu(out)
        return out


class ResNetPullBlock(nn.Module):
    def __init__(self, kernel_size=(2,2), stride=2, padding=1, p=0.5):
        super().__init__()
        self.seq = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Dropout(p=p),
            )

    def forward(self, x):
        out = self.seq(x)
        return out
