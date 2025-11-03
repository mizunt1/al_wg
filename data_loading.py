import torch
from torchvision import transforms
import random
from waterbirds_dataset import WaterbirdsDataset
from torchvision import transforms
from celeba import CelebA
import numpy as np
import collections

def waterbirds(num_minority_points, num_majority_points,
               metadata_path='metadata_larger.csv', root_dir='data/', img_size=512):
    use_cuda = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if img_size == None:
        img_size = 512
        
    trans = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    # training datasets
    split_scheme = {"wl_train":0, "lw_train": 1, "ww_train": 2, "ll_train": 3,
                    "wl_test":4, "lw_test": 5, "ww_test": 6, "ll_test": 7, "val": 8}
    split_names = {"wl_train":"wl_train", "ww_train": 'ww_train', "ll_train": 'll_train',
                   "lw_train": "lw_train","wl_test": 'wl_test',
                   "ww_test":"ww_test", "ll_test": "ll_test", "lw_test": "lw_test",
                   "val": "val"}

    dataset = WaterbirdsDataset(version='larger', root_dir=root_dir, download=True,
                                split_scheme=split_scheme, split_names=split_names,
                                metadata_name=metadata_path, use_rep=False)
    num_wl_points = num_minority_points //2
    num_lw_points = num_minority_points //2
    num_ww_points = num_majority_points //2
    num_ll_points = num_majority_points //2

    trainingwl_data = dataset.get_subset("wl_train", transform=trans, num_points=num_wl_points)
    traininglw_data = dataset.get_subset("lw_train", num_points=num_lw_points, transform=trans)
    trainingww_data = dataset.get_subset("ww_train", num_points=num_ww_points, transform=trans)
    trainingll_data = dataset.get_subset("ll_train", num_points=num_ll_points, transform=trans)
    print(f"Training data used sizes wl : {len(trainingwl_data)}, lw : {traininglw_data}, ww: {trainingww_data}, ll: {trainingll_data}")

    ww_test = dataset.get_subset("ww_test", transform=trans)
    wl_test = dataset.get_subset("wl_test", transform=trans)
    ll_test = dataset.get_subset("ll_test", transform=trans)
    lw_test = dataset.get_subset("lw_test", transform=trans)

    val_data = dataset.get_subset("val", transform=trans)

    training_data_dict = {'wl_train': trainingwl_data, 'lw_train': traininglw_data,
                          'ww_train': trainingww_data, 'll_train': trainingll_data}
    test_data_dict = {'ww_test': ww_test, 'll_test': ll_test,
                      'lw_test': lw_test, 'wl_test': wl_test, 'val': val_data}
    return dataset, training_data_dict, test_data_dict

def waterbirds_n_sources(num_minority_points, num_majority_points, n_maj_sources=3,
               metadata_path='metadata_larger.csv', root_dir='data/', img_size=None):
    use_cuda = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if img_size == None:
        img_size = 512
        
    trans = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    # training datasets
    split_scheme = {"wl_train":0, "lw_train": 1, "ww_train": 2, "ll_train": 3,
                    "wl_test":4, "lw_test": 5, "ww_test": 6, "ll_test": 7, "val": 8}
    split_names = {"wl_train":"wl_train", "ww_train": 'ww_train', "ll_train": 'll_train',
                   "lw_train": "lw_train","wl_test": 'wl_test',
                   "ww_test":"ww_test", "ll_test": "ll_test", "lw_test": "lw_test",
                   "val": "val"}

    dataset = WaterbirdsDataset(version='larger', root_dir=root_dir, download=True,
                                split_scheme=split_scheme, split_names=split_names,
                                metadata_name=metadata_path, use_rep=False)
    testww_data = dataset.get_subset("ww_test", transform=trans)
    testwl_data = dataset.get_subset("wl_test", transform=trans)
    testll_data = dataset.get_subset("ll_test", transform=trans)
    testlw_data = dataset.get_subset("lw_test", transform=trans)
    val_data = dataset.get_subset("val", transform=trans)
    ##### training data min group #####
    num_wl_points = num_minority_points //2
    num_lw_points = num_minority_points //2
    
    num_ww_points = num_majority_points //2
    num_ll_points = num_majority_points //2
    num_ww_points_per_group = num_ww_points //(n_maj_sources)
    num_ll_points_per_group = num_ll_points //(n_maj_sources)

    trainingwl_data = dataset.get_subset("wl_train", num_points=num_wl_points, transform=trans, source_id=0)
    traininglw_data = dataset.get_subset("lw_train", num_points=num_lw_points, transform=trans, source_id=0)
    s0 = torch.utils.data.ConcatDataset([trainingwl_data, traininglw_data])

    data_sources = collections.defaultdict()
    data_sources[0] = s0
    ##### training data maj group ######
    max_ww_points = dataset.get_subset_max_size("ww_train")
    max_ll_points = dataset.get_subset_max_size("ll_train")

    ww_idxs = np.random.permutation([i for i in range(0, max_ww_points)])
    ll_idxs = np.random.permutation([i for i in range(0, max_ll_points)])
    for i in range(1, n_maj_sources+1):
        indxs_to_sample_ww = ww_idxs[int(i*(num_ww_points_per_group)):int((i+1)*(num_ww_points_per_group)) ]
        indxs_to_sample_ll = ll_idxs[int(i*(num_ww_points_per_group)):int((i+1)*(num_ww_points_per_group)) ]
        ww_data_one_group = dataset.get_subset("ww_train", sample_idx=indxs_to_sample_ww, source_id=i, transform=trans)
        ll_data_one_group = dataset.get_subset("ll_train", sample_idx=indxs_to_sample_ll, source_id=i, transform=trans)
        one_source = torch.utils.data.ConcatDataset([ww_data_one_group, ll_data_one_group])
        data_sources[i] = one_source
    test_data_dict = {'ww_test': testww_data, 'll_test': testll_data,
                      'lw_test': testlw_data, 'wl_test': testwl_data, 'val': val_data}
    return dataset, data_sources, test_data_dict


def celeba(num_minority_points, num_majority_points, root_dir='/tmp/', img_size=None):
    # Note that minority group is blond male and non blond female
    # celeba dataset must be moved with the following command to /tmp/
    # cp -r /network/scratch/m/mizu.nishikawa-toomey/celeba /tmp/
    if img_size != None:
        trans = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.PILToTensor()])
    else:
        trans = transforms.Compose([transforms.PILToTensor()])
    blond_male = CelebA(root_dir, download=True, transform=trans, split='train_mb')
    blond_female = CelebA(root_dir, download=True, transform=trans, split='train_fb')
    notblond_male = CelebA(root_dir, download=True, transform=trans, split='train_mnb')
    notblond_female = CelebA(root_dir, download=True, transform=trans, split='train_fnb')

    blond_male_test = CelebA(root_dir, download=True, transform=trans, split='test_mb')
    blond_female_test = CelebA(root_dir, download=True, transform=trans, split='test_fnb')
    notblond_male_test = CelebA(root_dir, download=True, transform=trans, split='test_mnb')
    notblond_female_test = CelebA(root_dir, download=True, transform=trans, split='test_fnb')
    val = CelebA(root_dir, download=True, transform=trans, split='valid')
    
    print(f"num male blond: {len(blond_male)}")
    print(f'num female blond: {len(blond_female)}')
    print(f'num male not blond : {len(notblond_male)}')
    print(f'num female not blond: {len(notblond_female)}')

    num_bm_points = int(num_minority_points /2)
    num_bf_points = int(num_majority_points /2)
    num_nbm_points = int(num_majority_points /2)
    num_nbf_points = int(num_minority_points /2)
    assert num_bm_points <= len(blond_male)
    assert num_bf_points <= len(blond_female)
    assert num_nbm_points <= len(notblond_male)
    assert num_nbf_points <= len(notblond_female)
    print(f"Training data used sizes mb : {num_bm_points}, fb : {num_bf_points}, mnb: {num_nbm_points}, fnb: {num_nbf_points}")
    idx_bm_points = random.sample([i for i in range(len(blond_male))], k=num_bm_points)
    idx_bf_points = random.sample([i for i in range(len(blond_female))], k=num_bf_points)
    idx_nbm_points = random.sample([i for i in range(len(notblond_male))], k=num_nbm_points)
    idx_nbf_points = random.sample([i for i in range(len(notblond_female))], k=num_nbf_points)
    data0 = torch.utils.data.Subset(blond_male, idx_bm_points)
    data1 = torch.utils.data.Subset(blond_female, idx_bf_points)
    data2 = torch.utils.data.Subset(notblond_male, idx_nbm_points)
    data3 = torch.utils.data.Subset(notblond_female, idx_nbf_points)
    training_data_dict = {'mb_train': data0, 'fb_train': data1, 'mnb_train': data2, 'fnb_train': data3}
    test_data_dict = {'mb_test': blond_male_test, 'fb_test': blond_female_test,
                      'mnb_test': notblond_male_test, 'fnb_test': notblond_female_test, 'val': val}

    return blond_male, training_data_dict, test_data_dict

def celeba_n_sources(num_minority_points, num_majority_points, n_maj_sources = 3, root_dir='/tmp/', img_size=None):
    #  minority group is mb fnb
    # celeba dataset must be moved with the following command to /tmp/
    # cp -r /network/scratch/m/mizu.nishikawa-toomey/celeba /tmp/
    if img_size != None:
        trans = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.PILToTensor()])
    else:
        trans = transforms.Compose([transforms.PILToTensor()])
    all_mb = CelebA(root_dir, download=True, transform=trans, split='train_mb')
    all_fb = CelebA(root_dir, download=True, transform=trans, split='train_fb')
    all_mnb = CelebA(root_dir, download=True, transform=trans, split='train_mnb')
    all_fnb = CelebA(root_dir, download=True, transform=trans, split='train_fnb')

    max_mb_points = len(all_mb)
    max_fb_points = len(all_fb)
    max_mnb_points = len(all_mnb)
    max_fnb_points = len(all_fnb)
    
    test_mb = CelebA(root_dir, download=True, transform=trans, split='test_mb')
    test_fb = CelebA(root_dir, download=True, transform=trans, split='test_fb')
    test_mnb = CelebA(root_dir, download=True, transform=trans, split='test_mnb')
    test_fnb = CelebA(root_dir, download=True, transform=trans, split='test_fnb')
    val = CelebA(root_dir, download=True, transform=trans, split='valid')
    
    # minority 
    num_mb_points_per_group = int(num_minority_points /2)
    num_fnb_points_per_group = int(num_minority_points /2)

    # majority
    num_fb_points_per_group = int(num_majority_points /n_maj_sources)
    num_mnb_points_per_group = int(num_majority_points /n_maj_sources)

    # majority
    fb_idxs = np.random.permutation([i for i in range(0, max_fb_points)])
    mnb_idxs = np.random.permutation([i for i in range(0, max_mnb_points)])

    # minority
    mb_idxs = np.random.permutation([i for i in range(0, num_mb_points_per_group)])
    fnb_idxs = np.random.permutation([i for i in range(0, num_fnb_points_per_group)])

    # minority
    mb = CelebA(root_dir, download=True, transform=trans, split='train_mb', sample_idx=mb_idxs, source_id=0)
    fnb = CelebA(root_dir, download=True, transform=trans, split='train_fnb', sample_idx=fnb_idxs, source_id=0)

    s0 = torch.utils.data.ConcatDataset([mb, fnb])

    data_sources = collections.defaultdict()
    data_sources[0] = s0
    for i in range(1, n_maj_sources+1):
        indxs_to_sample_mnb = mnb_idxs[int(i*(num_mnb_points_per_group)):int((i+1)*(num_mnb_points_per_group)) ]
        indxs_to_sample_fb = fb_idxs[int(i*(num_fb_points_per_group)):int((i+1)*(num_fb_points_per_group)) ]
        mnb_data_one_group = CelebA(root_dir, download=True, transform=trans, split='train_mnb',
                                    sample_idx=indxs_to_sample_mnb, source_id=i)
        fb_data_one_group = CelebA(root_dir, download=True, transform=trans, split='train_fb',
                                    sample_idx=indxs_to_sample_fb, source_id=i)
        one_source = torch.utils.data.ConcatDataset([fb_data_one_group, mnb_data_one_group])
        data_sources[i] = one_source
    test_data_dict = {'mb_test': test_mb, 'fb_test': test_fb,
                      'mnb_test': test_mnb, 'fnb_test': test_fnb, 'val': val}
    return test_mb, data_sources, test_data_dict

def celeba_non_sp_load(num_minority_points, num_majority_points, batch_size, root_dir='/tmp/', img_size=None):
    # Note that minority group is blond male and non blond female
    if img_size != None:
        trans = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.PILToTensor()])
    else:
        trans = transforms.Compose([transforms.PILToTensor()])

    blond_male = CelebA(root_dir, download=True, transform=trans, split='train_bm')
    blond_female = CelebA(root_dir, download=True, transform=trans, split='train_bf')
    notblond_male = CelebA(root_dir, download=True, transform=trans, split='train_nbm')
    notblond_female = CelebA(root_dir, download=True, transform=trans, split='train_nbf')

    blond_male_test = CelebA(root_dir, download=True, transform=trans, split='test_bm')
    blond_female_test = CelebA(root_dir, download=True, transform=trans, split='test_bf')
    notblond_male_test = CelebA(root_dir, download=True, transform=trans, split='test_nbm')
    notblond_female_test = CelebA(root_dir, download=True, transform=trans, split='test_nbf')
    val = CelebA(root_dir, download=True, transform=trans, split='valid')
    
    print(f"num male blond: {len(blond_male)}")
    print(f'num female blond: {len(blond_female)}')
    print(f'num male not blond : {len(notblond_male)}')
    print(f'num female not blond: {len(notblond_female)}')
    max_points_female = len(notblond_female)
    max_points_male = len(blond_male)

    num_bm_points = int(num_minority_points /2)
    num_nbm_points = int(num_minority_points /2)
    num_bf_points = int(num_majority_points /2)
    num_nbf_points = int(num_majority_points /2)
    assert num_bm_points <= max_points_male
    assert num_bf_points <= max_points_female
    assert num_nbm_points <= max_points_male
    assert num_nbf_points <= max_points_female
    
    idx_bm_points = random.sample([i for i in range(len(blond_male))], k=num_bm_points)
    idx_bf_points = random.sample([i for i in range(len(blond_female))], k=num_bf_points)
    idx_nbm_points = random.sample([i for i in range(len(notblond_male))], k=num_nbm_points)
    idx_nbf_points = random.sample([i for i in range(len(notblond_female))], k=num_nbf_points)

    data0 = torch.utils.data.Subset(blond_male, idx_bm_points)
    data1 = torch.utils.data.Subset(blond_female, idx_bf_points)
    data2 = torch.utils.data.Subset(notblond_male, idx_nbm_points)
    data3 = torch.utils.data.Subset(notblond_female, idx_nbf_points)
    print(f"Training data used sizes mb : {num_bm_points}, fb : {num_bf_points}, mnb: {num_nbm_points}, fnb: {num_nbf_points}")
    training_data_dict = {'mb_train': data0, 'fb_train': data1, 'mnb_train': data2, 'fnb_train': data3}
    training_data = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([data0, data1, data2, data3]), shuffle=True, batch_size=batch_size)
    test_data_dict = {'mb_test': blond_male_test, 'fb_test': blond_female_test,
                      'mnb_test': notblond_male_test, 'fnb_test': notblond_female_test, 'val': val}
    test_data = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([blond_male_test, blond_female_test, notblond_male_test, notblond_female_test]),
        shuffle=True, batch_size=batch_size)

    return training_data, test_data, training_data_dict, test_data_dict

def cmnist_n_sources(num_minority_points, num_majority_points, n_maj_sources):
    trans = transforms.Compose([transforms.ToTensor()])
    start_idx = 0
    data_sources = collections.defaultdict()

    dataset = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                              causal_noise=0,
                              transform=trans, start_idx=start_idx, num_samples=num_minority_points, 
                              red=0, source_id=0)
    data_sources[0] = dataset
    start_idx += num_minority_points
    num_majority_points_per_group = num_majority_points // n_maj_sources
    for i in range(1, n_maj_sources + 1):
        dataset = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                  causal_noise=0,
                                  transform=trans, start_idx=start_idx, num_samples=num_majority_points_per_group, 
                                  red=1, group_idx=i)
        start_idx += num_majority_points_per_group
        data_souces[i] = dataset
    dataset0_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                      causal_noise=0,
                                      transform=trans, start_idx=start_idx, num_samples=5000,
                                      source_id=0, red=0)
    start_idx += 5000
    dataset1_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                      causal_noise=0,
                                      transform=trans, start_idx=start_idx, num_samples=5000,
                                      source_id=1, red=1)

        
    return dataset, data_sources, {'y0r': data0_unseen, 'y1r': data1_unseen}


if __name__ == "__main__":
    training_data, test_data, training_data_dict = celebA(100, 1000, 20)
