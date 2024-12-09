import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft as fft


class AdaptiveFilterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Filtered_X_acc, X_ppg):
        pred_fft = fft.fft(Filtered_X_acc.to(dtype = torch.complex128), dim=-1)
        true_fft = fft.fft(X_ppg.to(dtype = torch.complex128), dim=-1)
        mse_loss = nn.MSELoss()
        return mse_loss(torch.abs(pred_fft), torch.abs(true_fft))
