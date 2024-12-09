import torch
import torch.nn as nn
import torch.fft as fft
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


def causal_padding(x, kernel_size):
    padding = kernel_size - 1
    return F.pad(x, (padding, 0))


class ConvolutionBlock(nn.Module):

    def __init__(
        self, num_filters, kernel_size=5, dilation_rate=2, pooling_kernel_size=2
    ):
        super(ConvolutionBlock, self).__init__()

        n_input_filters, n_output_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        self.conv1 = nn.Conv1d(
            n_input_filters,
            n_output_filters,
            kernel_size=self.kernel_size,
            dilation=dilation_rate,
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            n_output_filters,
            n_output_filters,
            kernel_size=self.kernel_size,
            dilation=dilation_rate,
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(
            n_output_filters,
            n_output_filters,
            kernel_size=self.kernel_size,
            dilation=dilation_rate,
        )
        self.relu3 = nn.ReLU()

        self.avg_pooling = nn.AvgPool1d(kernel_size=pooling_kernel_size)
        self.dropout = nn.Dropout(p=0.5)

    def _causal_padding(self, x, kernel_size):
        # padding = kernel_size - 1
        padding = self.dilation_rate * (kernel_size - 1)  # Left-side padding
        return F.pad(x, (padding, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self._causal_padding(x, self.kernel_size))
        x = self.relu1(x)
        x = self.conv2(self._causal_padding(x, self.kernel_size))
        x = self.relu2(x)
        x = self.conv3(self._causal_padding(x, self.kernel_size))
        x = self.relu3(x)
        x = self.avg_pooling(x)
        x = self.dropout(x)
        return x


class KID_PPG(nn.Module):
    def __init__(self, input_shape, include_attention_weights=False):
        super(KID_PPG, self).__init__()
        self.input_shape = input_shape
        self.include_attention_weights = include_attention_weights
        self.conv_block1 = ConvolutionBlock((1, 32), pooling_kernel_size=4)
        self.conv_block2 = ConvolutionBlock((32, 48))
        self.conv_block3 = ConvolutionBlock((48, 64))
        self.attention = nn.MultiheadAttention(
            embed_dim=16,
            num_heads=4,
        )
        self.layer_norm = nn.LayerNorm(16)
        self.flatten_layer = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.dropout = nn.Dropout(p=0.125)
        self.fc2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x_bvp_t: torch.Tensor, x_bvp_t_1: torch.Tensor) -> torch.Tensor:
        x_bvp_t = self.conv_block1(x_bvp_t)
        x_bvp_t_1 = self.conv_block1(x_bvp_t_1)

        x_bvp_t = self.conv_block2(x_bvp_t)
        x_bvp_t_1 = self.conv_block2(x_bvp_t_1)

        x_bvp_t = self.conv_block3(x_bvp_t)
        x_bvp_t_1 = self.conv_block3(x_bvp_t_1)

        query, key, value = (
            x_bvp_t,
            x_bvp_t_1,
            x_bvp_t_1,
        )  # query is the present time step bvp, key and value are the previous time step's bvp
        x, attention_weights = self.attention(query, key, value)
        x = x + x_bvp_t

        x = self.layer_norm(x)

        x = self.flatten_layer(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if self.include_attention_weights:
            return x, attention_weights
        else:
            return x
