#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self, embed_size, kernel_size=5):
        super(CNN, self).__init__()
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(50, embed_size, kernel_size)

    def forward(self, x_reshaped):
        """
        Args:
            x_reshaped: tensor (batch_size, embed_size, word_len)
        
        Returns:
            x_convout: tensor (batch_size, embed_size)
        """
        x_conv = F.relu(self.conv1d(x_reshaped))  # (batch_size, embed_size, word_len - kernel_size + 1)
        x_convout = torch.max(x_conv, dim=2).values  # (batch_size, embed_size)
        return x_convout

### END YOUR CODE

