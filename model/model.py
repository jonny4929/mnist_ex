import torch
from torch._C import set_flush_denormal
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules import batchnorm

vgg_arg={
    'vgg11':[1,1,2,2,2],
    'vgg13':[2,2,2,2,2],
    'vgg16':[2,2,3,3,3],
    'vgg19':[2,2,4,4,4]
}

resnet_arg={
    'resnet18':[2,2,2,2],
    'resnet34':[3,4,6,3],
    'resnet50':[3,4,6,3],
    'resnet101':[3,4,23,3],
    'resnet152':[3,8,36,3]
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
        kwargs['bias']=False
        block_list=[
            nn.Conv2d(*args,**kwargs)
        ]
        if batch_norm is not None:
            block_list.append(batch_norm(block_list[0].out_channels))
        if act is not None:
            block_list.append(act())
        self.layer=nn.Sequential(*block_list)
    
    def forward(self,inputs):
        return self.layer(inputs)


class LeNet(nn.Module):
    def __init__(self,act=nn.ReLU,batch_norm=nn.BatchNorm2d):
        """
        docstring
        """
        super(LeNet,self).__init__()
        self.feature=nn.Sequential(
            basic_conv(3,6,5,padding=2,act=act,batch_norm=batch_norm),
            nn.MaxPool2d(2,2),
            basic_conv(6,10,5,act=act,batch_norm=batch_norm),
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
    def  __init__(self,name='vgg16',act=nn.ReLU,batch_norm=nn.BatchNorm2d):
        """
        docstring
        """
        super(VggNet,self).__init__()
        self.padding=nn.ZeroPad2d(2)
        arg=vgg_arg[name]
        in_channel=1
        for i,num in enumerate(arg):
            layer_list=[]
            for _ in range(num):
                layer_list.append(basic_conv(in_channel,2**(i+6),3,padding=1,act=act,batch_norm=batch_norm))
                in_channel=2**3
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
        feature=self.features(inputs)
        return self.classifier(feature)

class ResNet(nn.Module):
    def __init__(self,name='resnet50',act=nn.ReLU,batch_norm=nn.BatchNorm2d):
        """
        docstring
        """
        super(ResNet,self).__init__()
        self.act=act
        self.batch_norm=batch_norm
        self.basic_block=get_resnet_block(name)
        self.layer_nums=resnet_arg[name]

        self.padding=nn.ZeroPad2d(2)
        self.layer1=basic_conv(1,64,7,padding=3,stride=2,act=act,batch_norm=batch_norm)
        block_list=[nn.MaxPool2d(3,2,padding=1)]
        self.input_channels=64
        for layer in range(len(self.layer_nums)):
            block_list+=self.make_layers(layer)
            setattr(self,'layer%d'%(layer+2),nn.Sequential(*block_list))
            block_list=[]
        self.classifier=nn.Sequential(
            nn.Linear(2048,256),
            act(),
            nn.Linear(256,256),
            act(),
            nn.Linear(256,10)
        )

    def make_layers(self,layer):
        """
        docstring
        """
        num=self.layer_nums[layer]
        block_list=[]
        for i in range(num):
            block_list.append(self.basic_block(self.input_channels,2**(layer+6),act=self.act,batch_norm=self.batch_norm))
            self.input_channels=2**(layer+8)
        return block_list

    def forward(self,inputs):
        """
        docstring
        """
        inputs=self.padding(inputs)
        for i in range(1,6):
            inputs=getattr(self,'layer%d'%i)(inputs)
        inputs=inputs.view(-1,2048)
        return self.classifier(inputs)

        

def get_resnet_block(name):
    """
    docstring
    """
    class resnet_block_2layers(nn.Module):
        def __init__(self,input_channels,hidden_channels,act=nn.ReLU,batch_norm=nn.BatchNorm2d):
            """
            docstring
            """
            super(resnet_block_2layers,self).__init__()
            if input_channels!=hidden_channels:
                stride=2
                self.downsample=basic_conv(input_channels,hidden_channels,3,stride=2,act=None,batch_norm=batch_norm)
            else:
                stride=1
                self.downsample=None
            self.layers=nn.Sequential(
                basic_conv(input_channels,hidden_channels,3,padding=1,stride=stride,act=act,batch_norm=batch_norm),
                basic_conv(hidden_channels,hidden_channels,3,padding=1,act=None,batch_norm=batch_norm)
            )

        def forward(self,inputs):
            """
            docstring
            """
            if self.downsample is None:
                return nn.functional.relu(self.layers(inputs)+inputs)
            return nn.functional.relu(self.layers(inputs)+self.downsample(inputs))

    class resnet_block_3layers(nn.Module):
        def __init__(self,input_channels,hidden_channels,act=nn.ReLU,batch_norm=nn.BatchNorm2d):
            """
            docstring
            """
            super(resnet_block_3layers,self).__init__()
            output_channels=hidden_channels*4
            if input_channels!=output_channels:
                stride=2
                if input_channels==64:stride=1
                self.downsample=basic_conv(input_channels,output_channels,1,stride=stride,act=None,batch_norm=batch_norm)
            else:
                stride=1
                self.downsample=None
            self.layers=nn.Sequential(
                basic_conv(input_channels,hidden_channels,1,stride=stride,act=None,batch_norm=batch_norm),
                basic_conv(hidden_channels,hidden_channels,3,padding=1,act=None,batch_norm=batch_norm),
                basic_conv(hidden_channels,output_channels,1,act=None,batch_norm=batch_norm)
            )


        def forward(self,inputs):
            """
            docstring
            """
            if self.downsample is None:
                return nn.functional.relu(self.layers(inputs)+inputs)
            return nn.functional.relu(self.layers(inputs)+self.downsample(inputs))
    
    if name in ['resnet18','resnet34']:
        return resnet_block_2layers
    elif name in ['resnet50','resnet101','resnet152']:
        return resnet_block_3layers
    else:
        return None
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
    def  __init__(self,name='vgg16',act=nn.ReLU,batch_norm=nn.BatchNorm2d):
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
                layer_list.append(basic_conv(in_channels,out_channels,3,act=act,batch_norm=batch_norm,padding=1))
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

import torchvision
class torchresnet(nn.Module):
    def  __init__(self,*args,**kwargs):
        super(torchresnet,self).__init__()
        self.padding=nn.ZeroPad2d(2)
        self.net=torchvision.models.resnet50()
        self.net.conv1=nn.Conv2d(1,64,7,stride=2,padding=3,bias=False)
        self.net.avgpool=nn.Sequential()
        self.net.fc=nn.Sequential()
        self.classifier=nn.Sequential(
            nn.Linear(2048,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
    def forward(self,inputs):
        """
        docstring
        """
        inputs=self.padding(inputs)
        inputs=self.net(inputs).view(-1,2048)
        return self.classifier(inputs)