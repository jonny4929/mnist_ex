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
class basic_conv(nn.Module):
    def  __init__(self,act=nn.ReLU,batch_norm=nn.BatchNorm2d,*args,**kwargs):
        """
        docstring
        """
        block_list=[
            nn.Conv2d(*args,**kwargs),
            act()
        ]
        if batch_norm is not None:
            block_list.insert(1,batch_norm(block_list[0].out_channels))
        return nn.Sequential(*block_list)

class LeNet(nn.Module):
    def __init__(self,act=nn.ReLU):
        """
        docstring
        """
        super(LeNet,self).__init__()
        self.feature=nn.Sequential(
            basic_conv(3,6,5,padding=2,act=act),
            nn.MaxPool2d(2,2),
            basic_conv(6,10,5,act=act),
            nn.MaxPool2d(2,2)
        )
        self.classifier=nn.Sequential(
            nn.Linear(16*5*5,120),
            act(),
            nn.Linear(120,84),
            act(),
            nn.Linear(84,10)
        )

    def forward(self,inputs):
        """
        docstring
        """
        x=self.feature(inputs).view(-1,16*5*5)
        return self.classifier(x)

