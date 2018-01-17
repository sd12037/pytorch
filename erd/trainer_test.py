import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys, os
sys.path.append(os.pardir)
from mymodule.trainer import Trainer
from mymodule.utils import data_loader, evaluator
from tensorboardX import SummaryWriter

writer = SummaryWriter()

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 200
batch_size = 100
learning_rate = 0.001

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 2048)
        self.linear2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.linear(x)
        out = self.linear2(out)
        return out

model = LogisticRegression(input_size, num_classes)
model.cuda()
# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

trainer = Trainer(model, criterion, optimizer,
                  train_loader, test_loader,
                  val_num=10, early_stopping=10,
                  writer=writer)
trainer.run(epochs=num_epochs)
