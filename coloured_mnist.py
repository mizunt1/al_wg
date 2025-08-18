import os
import shutil

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import torch.nn as nn

from models import CMLineardo
from tools import entropy_drop_out

def color_grayscale_arr(arr, red=True, flip_colours=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if flip_colours:
        if red:
            arr = np.concatenate([arr,
                                  np.zeros((h, w, 2), dtype=dtype)], axis=2)
        else:
            arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                  arr,
                                  np.zeros((h, w, 1), dtype=dtype)], axis=2)
    else:
        if red:
            arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                  arr,
                                  np.zeros((h, w, 1), dtype=dtype)], axis=2)
        else:
            arr = np.concatenate([arr,
                                  np.zeros((h, w, 2), dtype=dtype)], axis=2)
    return arr

class ColoredMNIST(datasets.VisionDataset):
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
    def __init__(self, root='./data', env='train1', transform=None, target_transform=None, noise=True, sub_data=None):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        self.noise = noise
        self.prepare_colored_mnist()
        if env in ['train1', 'train2', 'test']:
            self.data_label_tuples = torch.load(
                os.path.join(self.root, 'ColoredMNIST', env) + '.pt', weights_only=False)
        elif env == 'all_train':
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt'), weights_only=False) + \
                torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'), weights_only=False)
        elif env == 'test':
            self.data_label_tuples = torch.load(
                os.path.join(self.root, 'ColoredMNIST', 'test.pt'), weights_only=False)

        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')
        if sub_data != None:
            self.data_label_tuples[sub_data[0]:sub_data[1]]

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
        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
            and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
            and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
            #print('Colored MNIST dataset already exists')
            return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train1_set = []
        train2_set = []
        test_set = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1
            
            # Flip label with 25% probability

            # Flip the color with a probability e that depends on the environment
            if idx < 20000:
                # causal noise in first environment
                if np.random.uniform() < 0.2:
                    binary_label = 1 - binary_label 
                # Color the image either red or green according to its possibly flipped label
                color_red = binary_label == 0
                # spurious noise in first env 
                if np.random.uniform() < 0:
                    color_red = not color_red
            elif idx < 40000:
                # causal noise in second environment
                if np.random.uniform() < 0.40:
                    binary_label = binary_label ^ 1
                color_red = binary_label == 0
                # spurious noise in second environment
                if np.random.uniform() < 0.0:
                    color_red = not color_red
            else:
                # spurious in test
                if np.random.uniform() < 0.5:
                    color_red = not color_red

            if idx < 20000:
                colored_arr = color_grayscale_arr(im_array, red=color_red, flip_colours=False)
                train1_set.append((Image.fromarray(colored_arr), binary_label))
            elif idx < 40000:
                colored_arr = color_grayscale_arr(im_array, red=color_red, flip_colours=True)
                train2_set.append((Image.fromarray(colored_arr), binary_label))
            else:
                test_set.append((Image.fromarray(colored_arr), binary_label))

        if os.path.exists(colored_mnist_dir):
            shutil.rmtree(colored_mnist_dir)
        os.makedirs(colored_mnist_dir)
        torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
        torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))


def test(model=None, device=None, test_loader=None, set_name="test set"):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        accs_info = {'red_y1_ttl': 0, 'red_y1_correct' : 0, 'red_y0_ttl' : 0,
                     'red_y0_correct' : 0, 'green_y1_ttl' : 0, 'green_y1_correct' : 0,
                     'green_y0_ttl': 0, 'green_y0_correct' : 0, 'total_correct':0, 'test_loss':0}

        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
            pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                               torch.Tensor([1.0]).to(device),
                               torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
            accs_info = accuracy_breakdown(data, target, pred, accs_info)
        test_log = {'test red_y1_ttl': accs_info['red_y1_ttl'],
                    'test red_y1_acc' : accs_info['red_y1_correct']/accs_info['red_y1_ttl'],
                    'test red_y0_ttl' :accs_info['red_y0_ttl'],
                    'test red_y0_acc' : accs_info['red_y0_correct']/accs_info['red_y0_ttl'],
                    'test green_y1_ttl' : accs_info['green_y1_ttl'],
                    'test green_y1_acc' : accs_info['green_y1_correct']/accs_info['green_y1_ttl'],
                    'test green_y0_ttl': accs_info['green_y0_ttl'],
                    'test green_y0_acc' : accs_info['green_y0_correct']/accs_info['green_y0_ttl'],
                    'test total_acc':accs_info['total_correct']/len(test_loader.dataset), 'test loss': test_loss}

    return test_log, accs_info


def train(model=None, device=None, train_loader=None, epochs=None):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    accs_info = {'red_y1_ttl': 0, 'red_y1_correct' : 0, 'red_y0_ttl' : 0, 'red_y0_correct' : 0, 'green_y1_ttl' : 0, 'green_y1_correct' : 0,
                 'green_y0_ttl': 0, 'green_y0_correct' : 0, 'total_correct':0}
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).float()
            optimizer.zero_grad()
            import pdb
            pdb.set_trace()
            output = model(data)
            pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                               torch.Tensor([1.0]).to(device),
                               torch.Tensor([0.0]).to(device))  # get the index of the max log-probability

            accs_info = accuracy_breakdown(data, target, pred, accs_info)

            loss = F.binary_cross_entropy_with_logits(output, target)

            loss.backward()
            optimizer.step()
        train_log = {'red_y1_ttl': accs_info['red_y1_ttl'],
                     'red_y1_acc' : accs_info['red_y1_correct']/accs_info['red_y1_ttl'],
                     'red_y0_ttl' :accs_info['red_y0_ttl'],
                     'red_y0_acc' : accs_info['red_y0_correct']/accs_info['red_y0_ttl'],
                     'green_y1_ttl' : accs_info['green_y1_ttl'],
                     'green_y1_acc' : accs_info['green_y1_correct']/accs_info['green_y1_ttl'],
                     'green_y0_ttl': accs_info['green_y0_ttl'],
                     'green_y0_correct' : accs_info['green_y0_correct']/accs_info['green_y0_ttl'],
                     'total_acc':accs_info['total_correct']/((epoch+1)*len(train_loader.dataset)),
                     'train_loss': loss.detach().cpu()}
        run.log(train_log)
    return train_log, accs_info

def find_colour(data):
    data = data[:,:2,:,:]
    normalised_zero = -0.4242
    sum_zeros = normalised_zero*data.shape[-1]*data.shape[-2]
    summed_data = data.sum(axis=3).sum(axis=2)
    colours = np.argmax(abs(summed_data - sum_zeros), axis=1)
    return colours

def accuracy_breakdown(data, target, preds, dict_):
    data = data.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    colours = find_colour(data)
    # calculate total counts
    red_y1 = np.intersect1d(np.where(colours==0), np.where(target==1))
    green_y1 = np.intersect1d(np.where(colours==1), np.where(target==1))
    red_y0 = np.intersect1d(np.where(colours==0), np.where(target==0))
    green_y0 = np.intersect1d(np.where(colours==1), np.where(target==0))
    # calculate accuracy
    correct = np.array([p==t for p,t in zip(preds.detach().cpu().numpy(), target)])
    dict_['red_y1_correct'] += sum(correct[red_y1])
    dict_['red_y0_correct'] += sum(correct[red_y0])
    dict_['green_y1_correct'] += sum(correct[green_y1])
    dict_['green_y0_correct'] += sum(correct[green_y0])

    dict_['red_y1_ttl'] += len(red_y1) 
    dict_['red_y0_ttl'] += len(red_y0) 
    dict_['green_y0_ttl'] += len(green_y0) 
    dict_['green_y1_ttl'] += len(green_y1) 
    dict_['total_correct'] += sum(correct) 
    return dict_

def train_and_test_erm():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    all_train_loader = torch.utils.data.DataLoader(
        ColoredMNIST(root='./data', env='all_train',
                    transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
      batch_size=64, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    ])),
        batch_size=1000, shuffle=True, **kwargs)
    model = CMLineardo().to(device)

    for epoch in range(1, 2):
        train(model, device, all_train_loader, epoch)
        test(model, device, all_train_loader, set_name='train set')
        test(model, device, test_loader)



if __name__ == "__main__":
    mnist = ColoredMNIST(env='train2')
    mnist.prepare_colored_mnist()
    train_and_test_erm()


    
    
    
