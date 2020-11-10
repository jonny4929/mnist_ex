import torch
import torch.nn as nn
class DNN(nn.Module):
    def __init__(self,act=nn.ReLU):
        super(DNN,self).__init__()
        self.layers=nn.Sequential(
                nn.Linear(28*28,1024),
                act(),
                nn.Linear(1024,512),
                act(),
                nn.Linear(512,128),
                act(),
                nn.Linear(128,10)
        )

    def forward(self,inputs):
        inputs=inputs.reshape(-1,28*28)
        return self.layers(inputs)