import torch
def calc_param(layer):
    count=0
    for param in layer.parameters():
        pcount=1
        for i in range(len(param.shape)):
            pcount*=param.shape[i]
        count+=pcount
    return count