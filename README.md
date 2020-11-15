# mnist_ex
A simple project on the MNIST dataset using deeplearning
## the networks supported
1. DNN
1. Lenet
2. Vggnet
3. Resnet

## Getting start
To train the model, please first make sure you have create two folders called datasets and checkpoints. Maybe using the following command in jupyter:
```bash
!mkdir checkpoints datasets
```

### Installation
The following packages are necessary:
- pytorch
- torchvision
- TensorboardX

If you want to run this project with you GPU(s), cudatoolkit is also needed.

### Train your models
you can first run the example script in jupyter:
```bash
!python train.py --model lenet --lr 0.0001 --epoch 20
```
To know more about this script using:
```
!python train.py -h
```
