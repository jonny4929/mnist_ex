import torch
import torch.nn as nn

vgg_arg={
    'vgg11':[1,1,2,2,2],
    'vgg13':[2,2,2,2,2],
    'vgg16':[2,2,3,3,3],
    'vgg19':[2,2,4,4,4]
}

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
    def  __init__(self,in_channels,out_channels,kernel_size,act=nn.ReLU,batch_norm=nn.BatchNorm2d,*args,**kwargs):
        """
        docstring
        """
        super(basic_conv,self).__init__()
        args=(in_channels,out_channels,kernel_size)+args
        block_list=[
            nn.Conv2d(*args,**kwargs),
            act()
        ]
        if batch_norm is not None:
            block_list.insert(1,batch_norm(block_list[0].out_channels))
        self.layer=nn.Sequential(*block_list)
    
    def forward(self,inputs):
        return self.layer(inputs)

class LeNet(nn.Module):
    def __init__(self,act=nn.ReLU):
        """
        docstring
        """
        super(LeNet,self).__init__()
        self.feature=nn.Sequential(
            basic_conv(1,6,5,padding=2,act=act),
            nn.MaxPool2d(2,2),
            basic_conv(6,16,5,act=act),
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


class VggNet(nn.Module):
    def  __init__(self,name='vgg16',act=nn.ReLU):
        """
        docstring
        """
        super(VggNet,self).__init__()
        self.padding=nn.ZeroPad2d(2)
        arg=vgg_arg[name]
        in_channels=1
        for i,num in enumerate(arg):
            layer_list=[]
            out_channels=min(512,2**(i+6))
            for _ in range(num):
                layer_list.append(basic_conv(in_channels,out_channels,3,padding=1))
                in_channels=out_channels
            layer_list.append(nn.MaxPool2d(2,2))
            setattr(self,"layer%d" %(i+1),nn.Sequential(*layer_list))
        self.features=nn.Sequential(*[getattr(self,"layer%d"%(i+1)) for i in range(5)])
        self.classifier=nn.Sequential(
            nn.Linear(512,256),
            act(),
            nn.Linear(256,256),
            act(),
            nn.Linear(256,10)
            )
    
    def forward(self,inputs):
        """
        docstring
        """
        inputs=self.padding(inputs)
        feature=self.features(inputs).view(-1,512)
        return self.classifier(feature)