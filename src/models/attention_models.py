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
    
    def __init__(self, input_shape, num_filters, kernel_size = 5, dilation_rate = 2, pooling_kernel_size = 2):
        super(ConvolutionBlock).__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=self.kernel_size, dilation=dilation_rate)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=self.kernel_size, dilation=dilation_rate)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(num_filters, num_filters, kernel_size=self.kernel_size, dilation=dilation_rate)
        self.relu3 = nn.ReLU()
        
        self.avg_pooling = nn.AvgPool1d(kernel_size=pooling_kernel_size)
        self.dropout = nn.Dropout(p = 0.5)
        
        
    def _causal_padding(self, x, kernel_size):
        padding = kernel_size - 1
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
    def __init__(self, input_shape, include_attention_weights):
        super(KID_PPG).__init__()
        input_num_channels, input_height, input_width = input_shape
        self.include_attention_weights = include_attention_weights
        self.conv_block1 = ConvolutionBlock(32, pooling_kernel_size=4)
        self.conv_block2 = ConvolutionBlock(48)
        self.conv_block3 = ConvolutionBlock(64)
        self.attention = nn.MultiheadAttention(emd_dim = 16, num_heads=4,)
        self.layer_norm = nn.LayerNorm()
        self.flatten_layer = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(in_features= input_num_channels * input_height * input_width, out_features=32)
        self.fc2 = nn.Linear(in_features= 32, out_features= 1)
        
    def forward(self, x_bvp_t: torch.Tensor, x_bvp_t_1: torch.Tensor) -> torch.Tensor:
        x_bvp_t = self.conv_block1(x_bvp_t)
        x_bvp_t_1 = self.conv_block1(x_bvp_t_1)
        
        x_bvp_t = self.conv_block2(x_bvp_t)
        x_bvp_t_1 = self.conv_block2(x_bvp_t_1)
        
        x_bvp_t = self.conv_block3(x_bvp_t)
        x_bvp_t_1 = self.conv_block3(x_bvp_t_1)
        
        
        query, key, value = x_bvp_t, x_bvp_t_1, x_bvp_t_1 # query is the present bvp and key and value are the previous time step's bvp
        x, attention_weights = self.attention(query, key, value)
        x = x + x_bvp_t
        
        x = self.flatten_layer(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        if self.include_attention_weights:
            return x, attention_weights
        else: 
            return x
        
        
        
        