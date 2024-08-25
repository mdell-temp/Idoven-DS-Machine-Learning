import torch
from torch.nn import Module, Conv1d, BatchNorm1d, ReLU, Sequential, AdaptiveAvgPool1d, Linear, Dropout
from torch.nn.functional import relu
from collections import OrderedDict
from typing import List

class ResidualBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        """Residual Block with a single convolutional layer and a skip connection.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = BatchNorm1d(out_channels)
        
        self.conv2 = Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, 
                            stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = BatchNorm1d(out_channels)

        self.shortcut = Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, 
                       stride=stride, bias=False),
                BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return relu(out)

class ResNet(Module):
    def __init__(self, num_classes: int, input_channels: int, block_layers: List[int], 
                 block_channels: List[int], kernel_sizes: List[int]):
        """ResNet-style fully convolutional network for time series classification.

        Args:
            num_classes (int): The number of output classes.
            input_channels (int): The number of input channels.
            block_layers (List[int]): List with the number of residual blocks in each block layer.
            block_channels (List[int]): List with the number of output channels per residual block.
            kernel_sizes (List[int]): List of kernel sizes for the convolutions in each block layer.
        """
        super(ResNet, self).__init__()

        self.in_channels = input_channels

        # Building the residual blocks
        self.layer1 = self._make_layer(block_channels[0], block_layers[0], kernel_sizes[0], stride=1)
        self.layer2 = self._make_layer(block_channels[1], block_layers[1], kernel_sizes[1], stride=2)
        self.layer3 = self._make_layer(block_channels[2], block_layers[2], kernel_sizes[2], stride=2)

        # Global pooling and fully connected layers
        self.global_pool = AdaptiveAvgPool1d(1)
        self.fc = Linear(block_channels[-1], num_classes)

        self.model_name = "ResNet"

    def _make_layer(self, out_channels: int, blocks: int, kernel_size: int, stride: int) -> Sequential:
        """Helper function to create a layer of residual blocks.

        Args:
            out_channels (int): Number of output channels for the blocks.
            blocks (int): Number of residual blocks.
            kernel_size (int): Convolutional kernel size.
            stride (int): Stride of the first convolution in the first block.

        Returns:
            Sequential: A layer containing the residual blocks.
        """
        strides = [stride] + [1]*(blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, kernel_size, stride))
            self.in_channels = out_channels
        return Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x


