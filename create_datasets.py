from cmnist_ram import ColoredMNISTRAM
from torch.utils.data import ConcatDataset
import random

def groups_to_env(trans,spurious_noise=0, causal_noise=0, num_envs=10):
    groups_list = []
    group_size = 100
    y0ag_num = 20
    y0ar_num = 1
    y1ar_num = 20
    y1ag_num = 1
    start_idx = 0
    envs_list = []
    #y=0, a=R min
    for i in range(y0ar_num):
        y0ar = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                         causal_noise=causal_noise,
                                         transform=trans, start_idx=start_idx, num_samples=group_size,
                                         red=1, group_idx=0, specified_class=0)
        start_idx += group_size
        groups_list.append(y0ar)

    #y=0,a=G maj
    for i in range(y0ag_num):    
        y0ag = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                               causal_noise=causal_noise,
                               transform=trans, start_idx=start_idx, num_samples=group_size,
                               red=0, group_idx=1, specified_class=0)
        start_idx += group_size
        groups_list.append(y0ag)

    #y=1,a=r maj
    for i in range(y1ar_num):
        y1ar = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                               causal_noise=causal_noise,
                               transform=trans, start_idx=start_idx,
                               num_samples=group_size, red=1,
                               group_idx=2, specified_class=1) 
        start_idx += group_size
        groups_list.append(y1ar)

    #y=1, a=G min
    for i in range(y1ag_num):
        y1ag = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                         causal_noise=causal_noise,
                                         transform=trans, start_idx=5000, num_samples=group_size,
                                         red=0, group_idx=3, specified_class=1)
        start_idx += group_size
        groups_list.append(y1ag)

    random.shuffle(groups_list)
    groups_to_envs = random.choices(range(0, num_envs), k=len(groups_list))
    for idx, env in enumerate(groups_to_envs):
        if idx < num_envs:
            envs_list.append([groups_list[idx]])
        else:
            envs_list[groups_to_envs[idx]].append(groups_list[idx])

    envs_list_concat = [ConcatDataset(item) for item in envs_list]
    return envs_list_concat, start_idx



def yxm_groups_cmnist(trans, spurious_noise=0, causal_noise=0):
    #y=0,a=G
    dataset0_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=0, num_samples=5000,
                                     red=0, group_idx=0, specified_class=0)
    #y=1,a=G
    dataset1_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=0,
                                     num_samples=200, red=0,
                                     group_idx=1, specified_class=1) 
    #y=0, a=R
    dataset2_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=5000, num_samples=200,
                                     red=1, group_idx=2, specified_class=0)

    #y=1, a=R
    dataset3_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=5000,
                                     num_samples=5000, red=1, group_idx=3) 

    start_idx = 10000
    return [dataset0_train, dataset1_train, dataset2_train, dataset3_train], start_idx

def five_groups_cmnist(spurious_noise, causal_noise, data1_size, data2_size, trans):
    dataset0_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=0, num_samples=data1_size, 
                                     red=1, group_idx=0)
    dataset1_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=5000, num_samples=data1_size, 
                                     red=1, group_idx=1)
    dataset2_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=10000, num_samples=data1_size, 
                                     red=1, group_idx=2)
    dataset3_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=15000, num_samples=data1_size, 
                                     red=1, group_idx=3)
    
    dataset4_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=20000, num_samples=data2_size, red=0, group_idx=4) 
    start_idx = 20000 + data2_size
    return [dataset0_train, dataset1_train, dataset2_train, dataset3_train, dataset4_train], start_idx


def two_groups_cmnist(spurious_noise, causal_noise, data1_size, data2_size, trans):
    dataset0_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=0, num_samples=data1_size,
                                     red=1, group_idx=0)
    dataset4_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=20000,
                                     num_samples=data2_size, red=0, group_idx=1) 
    start_idx = 20000 + data2_size
    return [dataset0_train, dataset4_train], start_idx

def ten_groups_cmnist(spurious_noise, causal_noise, data1_size, data2_size, trans):
    data = []
    for i in range(9):
        start_idx = 0
        dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                  causal_noise=causal_noise,
                                  transform=trans, start_idx=start_idx, num_samples=data1_size, 
                                  red=1, group_idx=i)
        start_idx += data1_size
        data.append(dataset)
    dataset9_train = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                     causal_noise=causal_noise,
                                     transform=trans, start_idx=start_idx,
                                     num_samples=data1_size, red=0, group_idx=9) 
    data.append(dataset9_train)
    start_idx += data1_size
    return data, start_idx

def ten_groups_cmnist_multiple_int(spurious_noise, causal_noise, data1_size, data2_size, trans):
    data = []
    for i in range(8):
        start_idx = 0
        dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                  causal_noise=causal_noise,
                                  transform=trans, start_idx=start_idx, num_samples=data1_size, 
                                  red=1, group_idx=i)
        start_idx += data1_size
        data.append(dataset)
    dataset = ColoredMNISTRAM(root='./data', spurious_noise=0.3, 
                              causal_noise=causal_noise,
                              transform=trans, start_idx=start_idx, num_samples=data1_size//2, 
                              red=0, group_idx=8)
    start_idx += data1_size//2
    data.append(dataset)
    
    dataset = ColoredMNISTRAM(root='./data', spurious_noise=0.2, 
                              causal_noise=causal_noise,
                              transform=trans, start_idx=start_idx, num_samples=data1_size//2, 
                              red=0, group_idx=9)
    start_idx += data1_size//2
    data.append(dataset)
    return data, start_idx


def one_balanced_cmnist(data1_size, num_spurious_groups, trans=None):
    data = []
    for i in range(num_spurious_groups):
        start_idx = 0
        dataset = ColoredMNISTRAM(root='./data', 
                                  transform=trans, start_idx=start_idx, num_samples=data1_size, 
                                  red=1, group_idx=i)
        start_idx += data1_size
        data.append(dataset)
        
    dataset = ColoredMNISTRAM(root='./data', 
                              transform=trans, start_idx=start_idx, num_samples=data1_size//2, 
                              red=0, group_idx=num_spurious_groups)
    dataset2 = ColoredMNISTRAM(root='./data', 
                              transform=trans, start_idx=start_idx, num_samples=data1_size//2, 
                               red=1, group_idx=num_spurious_groups)

    data_balanced = ConcatDataset([dataset, dataset2])
    start_idx += data1_size
    data.append(data_balanced)
    return data, start_idx

def leaky_groups(data1_size, spurious_noise=0.05, num_spurious_groups=5, trans=None):
    data = []
    for i in range(num_spurious_groups):
        start_idx = 0
        dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                  transform=trans, start_idx=start_idx, num_samples=data1_size, 
                                  red=1, group_idx=i)
        start_idx += data1_size
        data.append(dataset)
        
    dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise,
                              transform=trans, start_idx=start_idx, num_samples=data1_size, 
                              red=0, group_idx=num_spurious_groups)
    start_idx += data1_size
    data.append(dataset)
    return data, start_idx

