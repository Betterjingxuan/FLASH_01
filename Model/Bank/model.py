import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(MLP, self).__init__()
        # Initialize the layers
        self.layers = nn.ModuleList()
        self.dropout_rate=dropout_rate
        
        # First layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.BatchNorm1d(hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Middle layers (9 layers)
        for _ in range(2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        # Apply each component in layers
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)
