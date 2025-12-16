import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_datasets=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_datasets=datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader=DataLoader(train_datasets,batch_size=32,shuffle=True)
test_loader=DataLoader(test_datasets,batch_size=32,shuffle=False)