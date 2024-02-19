import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, hidden_layers, num_classes, dropout_rate):
        super(MLPClassifier, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer size
        prev_size = 512
        for layer_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, layer_size))
            self.layers.append(nn.BatchNorm1d(layer_size))  
            self.layers.append(nn.ReLU())  
            self.layers.append(nn.Dropout(dropout_rate))  
            prev_size = layer_size
        
        # Output layer
        self.out = nn.Linear(prev_size, num_classes)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x


