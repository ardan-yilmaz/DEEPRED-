import torch
from torch.utils import data
import pandas as pd
import torch.nn as nn
import os
import io
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class DEEPRedDataset(Dataset):
    def __init__(self, data_path, labels_path):

        df_data = pd.read_csv(data_path, header=None) 
        sc = StandardScaler()
        x = sc.fit_transform(df_data.values)
        self.x = torch.tensor(x)       


        labels = pd.read_csv(labels_path, header=None) 
        self.y = torch.tensor(labels.values)        
                           
    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx, :]

    def __len__(self):
        return len(self.x)



class TwoLayer(nn.Module):
    def __init__(self, num_GO_Terms, num_layer1=1024, num_layer2=512, dropout=0.2):
        super().__init__()
        #layers of the nnet
        self.layer1 = nn.Linear(in_features=400, out_features=num_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_layer1)

        self.layer2 = nn.Linear(in_features=num_layer1, out_features=num_layer2)
        self.batchnorm2 = nn.BatchNorm1d(num_layer2)

        self.layer3 = nn.Linear(in_features=num_layer2, out_features=num_GO_Terms)

        self.dropout = nn.Dropout(p=dropout)
        self.out = torch.nn.Sigmoid()
        #self.num_GO_Terms = num_GO_Terms



    
    def forward(self, x):

      #hidden layer 1
      x = self.layer1(x)
      x = self.batchnorm1(x)
      self.act_func1 = nn.functional.relu(x)
      x = self.dropout(x)

      #hidden layer 2
      x = self.layer2(x)
      x = self.batchnorm2(x)
      self.act_func1 = nn.functional.relu(x)
      x = self.dropout(x)  

      x = self.layer3(x)    
      x = self.out(x)

      return x 




