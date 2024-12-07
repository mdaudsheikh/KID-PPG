import torch
import torch.nn as nn
import torch.fft as fft
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MotionArtifactSeparator(nn.Module):
    """Linear two-layer convolutional network for estimating motion artifacts."""
    def __init__(self):
        super(MotionArtifactSeparator, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 21), padding='same')  # Spatio-temporal filtering
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(3, 1), padding = 'valid')  # Merge channels

    def forward(self, X_acc):
        x = self.conv1(X_acc)
        x = self.conv2(x)
        return x