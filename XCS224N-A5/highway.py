#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn
import torch.nn.functional as F



class Highway(nn.Module):
    def __init__(self, embed_size):
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.w_proj = nn.Linear(embed_size, embed_size)
        nn.init.xavier_uniform_(self.w_proj.weight)
        self.w_gate = nn.Linear(embed_size, embed_size)
        nn.init.xavier_uniform_(self.w_gate.weight)

    def forward(self, x_convout):
        """
        Args:
            x_convout: tensor (batch_size, embed_size)

        Returns:
            x_highway: tensor (batch_size, embed_size)
        """
        x_proj = F.relu(self.w_proj(x_convout))
        x_gate = torch.sigmoid(self.w_gate(x_convout))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_gate
        return x_highway
### END YOUR CODE 

