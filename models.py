import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch 
import mc_dropout
import torchvision

def resnet50(classes=2, drop_out=0.0):
    model = torchvision.models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    d = model.fc.in_features
    model.fc = nn.Linear(d, classes)
    return model

def resnet50_plus(classes=2, drop_out=0.0):
    model = torchvision.models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    d = model.fc.in_features
    model.fc = nn.Linear(d, 256)
    extra = nn.Linear(256, classes)
    model.layer4.append(extra)
    return model

def wideresnet50(classes=2, drop_out=0.0):
    model = torchvision.models.wide_resnet50_2(pretrained=True)
    #for param in model.parameters():
    #    param.requires_grad = False
    d = model.fc.in_features
    model.fc = nn.Linear(d, classes)
    return model


class ConvNet(nn.Module):
    def __init__(self):
      super(ConvNet, self).__init__()
      self.conv1 = nn.Conv2d(3, 20, 20, 1)
      self.conv2 = nn.Conv2d(20, 50, 10, 1)
      self.conv3 = nn.Conv2d(50, 20, 5, 2)
      self.conv4 = nn.Conv2d(20, 10, 5, 2)
      self.fc1 = nn.Linear(10 * 5 * 5, 250)
      self.fc2 = nn.Linear(250, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)    
        x = x.view(-1, 10*5*5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class BayesianNet(mc_dropout.BayesianModule):
    # https://github.com/BlackHC/BatchBALD/blob/master/src/mnist_model.py
    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv1_drop = mc_dropout.MCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = mc_dropout.MCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = mc_dropout.MCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)
        return input

class BayesianNetRes50(mc_dropout.BayesianModule):
    # https://github.com/BlackHC/BatchBALD/blob/master/src/mnist_model.py
    def __init__(self, num_classes):
        super().__init__(num_classes)
        inner_rep = 256
        self.model = torchvision.models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, inner_rep)
        self.classifier = nn.Sequential(
            mc_dropout.MCDropout(),
            nn.Linear(inner_rep, num_classes)
        )

    def deterministic_forward_impl(self, x):
        x = self.model(x)
        return x
    
    def mc_forward_impl(self, input):
        input = self.classifier(input)
        return input

class Linear(nn.Module):
    def __init__(self, input_size, classes=4):
        super(Linear, self).__init__()
        self.layer = nn.Linear(input_size, 10)
        self.layer2 = nn.Linear(10, classes)

    def forward(self, input):
        out = self.layer(input)
        out = F.relu(out)
        out = self.layer2(out)
        return out

class Lineardo(nn.Module):
    def __init__(self, input_size, classes=4):
        super(Lineardo, self).__init__()
        self.layer = nn.Linear(input_size, 5)
        self.layer2 = nn.Linear(5, classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, input):
        out = self.layer(input)
        out = F.relu(out)
        out = self.drop(out)
        out = F.relu(out)
        out = self.layer2(out)
        return out

class CMLineardo(nn.Module):
    def __init__(self, input_size, classes=2, p=0.2):
        super(CMLineardo, self).__init__()
        self.layer = nn.Linear(input_size, 25)
        self.layer2 = nn.Linear(25, classes)
        self.drop = nn.Dropout(p)

    def forward(self, input):
        out = self.layer(input)
        out = F.relu(out)
        out = self.drop(out)
        out = self.layer2(out)
        return out

def erm(model, data, label, transforms, num_epochs=140, lr=1e-2,
        log_interval=5, weight_classes=True):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if weight_classes:
        _, counts = np.unique(label, return_counts=True)
        weight = torch.from_numpy(1/counts.astype('float32'))
    else:
        weight = None
    loss_fn = nn.CrossEntropyLoss(weight=weight) 
    #data,target = torch.stack(data), torch.stack(target)
    #data, target = data.to(device), target.to(device)
    data = transforms(data).squeeze(0)
    for epoch in range(num_epochs):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss_fn(output, label) #+ l1_norm*0.05
        loss.backward()
        optimizer.step()
        correct = (output.argmax(axis=1) == label).sum()/len(label)
        if epoch % log_interval == 0:
            print('Train epoch: {} Loss: {:.3f} correct prop train: {:.3f}' .format(
                epoch, loss.item(), correct))
    return correct, model

def batch_erm(model, data, label, transforms, num_epochs=140, lr=1e-2,
        log_interval=5, weight_classes=True):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if weight_classes:
        _, counts = np.unique(label, return_counts=True)
        weight = torch.from_numpy(1/counts.astype('float32'))
    else:
        weight = None
    loss_fn = nn.CrossEntropyLoss(weight=weight) 
    #data,target = torch.stack(data), torch.stack(target)
    #data, target = data.to(device), target.to(device)
    data = transforms(data).squeeze(0)
    for epoch in range(num_epochs):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss_fn(output, label) #+ l1_norm*0.05
        loss.backward()
        optimizer.step()
        correct = (output.argmax(axis=1) == label).sum()/len(label)
        if epoch % log_interval == 0:
            print('Train epoch: {} Loss: {:.3f} correct prop train: {:.3f}' .format(
                epoch, loss.item(), correct))
    return correct, model


def test(model, data, label, transforms):
    #data, target = data.to(device), target.to(device)
    # data, target = torch.FloatTensor(data).to(device), torch.FloatTensor(target).to(device)
    data = transforms(data).squeeze(0)
    model.eval()
    output = model(data)
    correct = (output.argmax(axis=1) == label).sum()/len(label)

    return correct
