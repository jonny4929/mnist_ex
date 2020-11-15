import torch,torchvision
from torch.nn.modules import loss
import torch.nn as nn
import torchvision.transforms as T
import argparse
import tensorboardX
from model.model import *

act_dict={
    'relu':nn.ReLU,
    'relu6':nn.ReLU6,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,
    'leakyrelu':nn.LeakyReLU,
}
model_dict={
    'dnn':(DNN,{}),
    'lenet':(LeNet,{}),
    'vggnet':(VggNet,{"name":'vgg16'}),
    'resnet':(ResNet,{}),
    'torchresnet':(torchresnet,{})
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
    parser.add_argument('--drop_rate', type=float, default=1)
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--model',type=str,default='dnn')
    parser.add_argument('--act',type=str,default='relu')
    parser.add_argument('--loss_func',type=str,default='crossentropy')
    parser.add_argument('--data_path',type=str,default='./datasets')
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--checkpoints_path',type=str,default='./checkpoints')
    return parser.parse_args()

def train(args):
    transformer=T.Compose([
    T.ToTensor(),
    T.Normalize((0.3081),(0.1307))
    ])
    train_data=torchvision.datasets.MNIST(root=args.data_path,transform=transformer,download=True,train=True)
    train_loader=torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=4)

    model_arg=model_dict[args.model][1]
    model_arg["act"]=(act_dict[args.act])
    device=torch.device(args.device)
    net=model_dict[args.model][0](**model_arg).to(device)

    if args.optimizer=='adam':
        optimizer=torch.optim.Adam(net.parameters(),lr=args.lr,betas=(0.9,0.99))
    elif args.optimizer=='SGD':
        optimizer=torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9)
    else:
        optimizer=None
    loss_func=loss_dict[args.loss_func]()
    
    writer=tensorboardX.SummaryWriter()

    current_acc=0
    for epoch in range(args.epoch):
        total_loss=0.
        total_acc=0.
        for i,(images,labels) in enumerate(train_loader):
            images,labels=images.to(device),labels.to(device)
            outputs=net(images)
            loss=loss_func(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            acc=torch.sum(outputs.argmax(-1)==labels).item()
            total_acc+=acc/args.batch_size

            writer.add_scalar('data/loss',loss,i+epoch*len(train_loader))
        print("epoch%3d: loss=%.4f ,acc:%.2f%% " %(epoch,total_loss/len(train_loader),total_acc*100/len(train_loader)))
        if(epoch%1==0):
            eval_acc=eval(args,net)
            writer.add_scalar('data/acc',eval_acc,epoch)
            if eval_acc>current_acc:
                torch.save(net.state_dict(),'%s/best_%s_model.pth' %(args.checkpoints_path,args.model))


def eval(args,net):
    """
    docstring
    """
    transformer=T.Compose([
    T.ToTensor(),
    T.Normalize((0.3081),(0.1307))
    ])
    device=torch.device(args.device)
    test_data=torchvision.datasets.MNIST(root=args.data_path,transform=transformer,train=False)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=args.batch_size,shuffle=True,drop_last=False)
    total_acc=0
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=net(images)
        acc=torch.sum(outputs.argmax(-1)==labels).item()
        total_acc+=acc
    print("eval acc=%.2f%%" %(total_acc/len(test_data)*100))
    return total_acc/len(test_data)    

if __name__ == '__main__':
    args=get_args()
    train(args)