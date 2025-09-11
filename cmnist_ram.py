import os

import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np
from PIL import Image
from torch.utils.data import Subset
import torch.optim as optim
import collections

def color_grayscale_arr(arr, red=True, flip_colours=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        # red appears in first channel
        arr = np.concatenate([arr,
                              np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        # green appears in second channel
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                              arr,
                              np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr


class ColoredMNISTRAM(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
        root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
        env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
        transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """
    def __init__(self, spurious_noise=0, causal_noise=0, train=True,
                 transform=None, num_samples=5000, start_idx=0,
                 add_digit=None, flip_sp=False, fiif=False, root='./data',
                 group_idx=-1, specified_class=None, red=1):
        super(ColoredMNISTRAM, self).__init__(root, 
                                              transform=transform)
        self.start_idx = start_idx
        self.num_samples = num_samples
        self.red = red
        self.causal_noise = causal_noise
        self.spurious_noise = spurious_noise
        self.train = train
        self.fiif = fiif
        self.specified_class = specified_class
        self.prepare_colored_mnist()
        self.add_digit = add_digit
        self.group_idx = group_idx
        
    def __getitem__(self, index):
        """
        Args:
        index (int): Index

        Returns:
        tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data_label_tuples[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.add_digit != None:
            img[0,0] = self.add_digit
        return {'data': img, 'target': target, 'group_id': self.group_idx}

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        train_mnist = datasets.mnist.MNIST(self.root, train=self.train, download=True)
        train_mnist = Subset(train_mnist, [i for i in range(self.start_idx, self.start_idx + self.num_samples)])
        dataset = []

        for idx, (im, label) in enumerate(train_mnist):
            im_array = np.array(im)
            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1
            color_red = binary_label == self.red
            if np.random.uniform() < self.causal_noise:
                binary_label = 1 - binary_label
            if not self.fiif:
                # if partially informative, the colour is downstream of 
                # noisy label
                color_red = binary_label == 0
            if np.random.uniform() < self.spurious_noise:
                color_red = not color_red
            colored_arr = color_grayscale_arr(im_array, red=color_red,)
            dataset.append((Image.fromarray(colored_arr), binary_label))
        if self.specified_class != None:
            self.data_label_tuples = [data for data in dataset if data[1] == self.specified_class]
        else:
            self.data_label_tuples = dataset

def train_batched(model=None, epochs=30, dataloader=None,
                  dataloader_test=None, lr=0.001, flatten=False, num_groups=2):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)
    groups = []
    group_dict = collections.defaultdict(int)
    for epoch in range(epochs):
        total_correct = 0
        total_points = 0
        for batch_idx, out_dict in enumerate(dataloader):
            data = out_dict['data']
            target = out_dict['target']
            group_id = out_dict['group_id']
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if flatten:
                data = data.reshape(-1, 3*28*28)
            output = model(data).squeeze(1)

            out = output.argmax(axis=1)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_correct += sum(out == target)
            total_points += len(target)
            if epoch == 0:
                groups.extend(group_id)
    total_correct_test = 0
    total_points_test = 0
    model.eval()
    for batch_idx, out_dict in enumerate(dataloader_test):
        data = out_dict['data']
        target = out_dict['target']

        data, target = data.to(device), target.to(device)
        if flatten:
            data = data.reshape(-1, 3*28*28)
        output = model(data).squeeze(1)
        out = output.argmax(axis=1)
        total_correct_test += sum(out == target)
        total_points_test += len(target)
    for i in range(num_groups):
        group_dict[i] = sum(np.array(groups) == i)
    return (total_correct / total_points).cpu().item(), (total_correct_test/total_points_test).cpu().item(), group_dict

def train_batched_weighted(model=None, epochs=30, dataloader=None, dataloader_test=None, lr=0.001):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_correct = 0
        total_points = 0
        for batch_idx, (data, target, _) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            data = data.reshape(-1, 3*28*28)
            output = model(data)
            out = output.argmax(axis=1)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_correct += sum(out == target)
            total_points += len(target)
    total_correct_test = 0
    total_points_test = 0
    model.eval()
    for batch_idx, (data, target, _) in enumerate(dataloader_test):
        data, target = data.to(device), target.to(device)
        data = data.reshape(-1, 3*28*28)
        output = model(data)
        out = output.argmax(axis=1)
        total_correct_test += sum(out == target)
        total_points_test += len(target)
    return (total_correct / total_points).cpu().item(), (total_correct_test/total_points_test).cpu().item()
    

def test_batched(model, dataloader_test):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    total_correct_test = 0
    total_points_test = 0
    model.eval()
    for batch_idx, out_dict in enumerate(dataloader_test):
        data = out_dict['data']
        target = out_dict['target']

        data, target = data.to(device), target.to(device)
        output = model(data).squeeze(1)
        out = output.argmax(axis=1)
        total_correct_test += sum(out == target)
        total_points_test += len(target)
    return (total_correct_test/total_points_test).cpu().item()

