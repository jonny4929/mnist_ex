import argparse
from os import name

import torch,torchvision
from torch.nn.modules import loss
from torch._C import device
import torch.nn as nn
import torchvision.transforms as T
from model.model import DNN

act_dict={
    'relu':nn.ReLU,
    'relu6':nn.ReLU6,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,
    'leakyrelu':nn.LeakyReLU,
}
model_dict={
    'dnn':DNN,
}
loss_dict={
    'crossentropy':nn.CrossEntropyLoss,
}
def get_args():
    """
    get arguments for this scripts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--drop_rate', type=float, default=0.97)
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--model',type=str,default='dnn')
    parser.add_argument('--act',type=str,default='relu')
    parser.add_argument('--loss_func',type=str,default='crossentropy')
    parser.add_argument('--data_path',type=str,default='./datasets')
    parser.add_argument('--device',type=str,default='cuda')
    return parser.parse_args()

def train(args):
    transformer=T.Compose([
    T.ToTensor(),
    T.Normalize((0.3081),(0.1307))
    ])
    train_data=torchvision.datasets.MNIST(root=args.data_path,transform=transformer,download=True,train=True)
    train_loader=torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,shuffle=True,drop_last=True)

    act=act_dict[args.act]
    device=torch.device(args.device)
    net=model_dict[args.model](act=act).to(device)

    if args.optimizer=='adam':
        optimizer=torch.optim.Adam(net.parameters(),lr=args.lr,betas=(0.9,0.99))
    elif args.optimizer=='SGD':
        optimizer=torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9)
    else:
        optimizer=None
    loss_func=loss_dict[args.loss_func]()
    
    current_acc=0
    for epoch in range(args.epoch):
        total_loss=0.
        total_acc=0.
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=net(images)

            loss=loss_func(outputs,labels)
            optimizer.zero_grads()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            acc=torch.sum(outputs.argmax(-1)==labels).item()
            total_acc+=acc/args.batch_size
        print("epoch%3d: loss=%.4f ,acc:%.2f%% " %(epoch,total_loss/len(train_loader),total_acc*100/len(train_loader)))
        if(epoch%2==1):
            eval_acc=eval(args,net)
            if eval_acc>current_acc:
                torch.save(net.state_dict(),'best_%s_model.pth' %args.model)


def eval(args,net):
    """
    docstring
    """
    transformer=T.Compose([
    T.ToTensor(),
    T.Normalize((0.3081),(0.1307))
    ])
    test_data=torchvision.datasets.MNIST(root=args.data_path,transform=transformer,train=False)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=args.batch_size,shuffle=True,drop_last=False)
    total_acc=0
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=net(images)
        acc=torch.sum(outputs.argmax(-1)==labels).item()
        total_acc+=acc
    return total_acc/len(test_data)    

if __name__ == '__main__':
    args=get_args()
    train(args)