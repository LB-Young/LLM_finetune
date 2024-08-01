"amp:自动混合精度训练，会降低现存的使用；"


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
print(torch.cuda.is_available())


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 10000)
        self.fc2 = nn.Linear(10000, 10000)
        self.fc3 = nn.Linear(10000, 2)
        # self.loss = nn.functional.cross_entropy(ignore_index=-100)

    def forward(self, x, y=None):
        x =self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        if y is not None:
            loss = nn.functional.cross_entropy(out.view(-1, 2), y.view(-1))
            return loss
        else:
            return out
        
device = 'cuda'
batch_size = 128
sequence_len = 200
embedding_dim = 10
num_epochs = 1
learning_ratet = 0.001
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_ratet)
def train(batch_size):
    try:
        for epoch in range(num_epochs):
            batch_x = torch.rand(batch_size, sequence_len, embedding_dim)
            batch_y = torch.randint(0,1,(batch_size, sequence_len, 1))
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = model(batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return
    except:
        print("common train batch_size:", batch_size)

def train_amp(batch_size):
    try:
        scaler = torch.cuda.amp.grad_scaler()
        for epoch in range(num_epochs):
            batch_x = torch.rand(batch_size, sequence_len, embedding_dim)
            batch_y = torch.randint(0,1,(batch_size, sequence_len, 1))
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # forward
            with torch.cuda.amp.autocast():
                loss = model(batch_x)
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # gradient descent
            scaler.step(optimizer)
            scaler.update()
        return
    except:
        print("amp train batch_size:", batch_size)

for batch_size in [4, 8, 16, 32, 64, 128, 256, 512]:
    train(batch_size)
    train_amp(batch_size)


def train_local(batch_size):
    for epoch in range(num_epochs):
        batch_x = torch.rand(batch_size, sequence_len, embedding_dim)
        batch_y = torch.randint(0,1,(batch_size, sequence_len, 1))
        batch_x = batch_x
        batch_y = batch_y
        loss = model(batch_x, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# train_local(2)