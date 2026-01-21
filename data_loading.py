import torch
from torchvision import transforms
import random
from waterbirds_dataset import WaterbirdsDataset
from iwildcam_dataset import IWildCamDataset
from torchvision import transforms
from camelyon17_dataset import Camelyon17Dataset
from cmnist_ram import ColoredMNISTRAM
from celeba import CelebA
import numpy as np
import collections
import math
from fmow_dataset import FMoWDataset

def waterbirds_n_sources(num_minority_points, num_majority_points, n_maj_sources=3,
               metadata_path='metadata_largerv2.csv', root_dir='data/', img_size=None):
    use_cuda = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if img_size == None:
        img_size = 512
        
    trans = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    # training datasets
    split_scheme = {"wl_train":0, "lw_train": 1, "ww_train": 2, "ll_train": 3,
                    "wl_val":4, "lw_val": 5, "ww_val": 6, "ll_val": 7,
                    "wl_test":8, "lw_test": 9, "ww_test": 10, "ll_test": 11}
    split_names = {key: key for key in split_scheme}

    dataset = WaterbirdsDataset(version='larger', root_dir=root_dir, download=True,
                                split_scheme=split_scheme, split_names=split_names,
                                metadata_name=metadata_path, use_rep=False)
    rng_state = np.random.get_state()
    valww_data = dataset.get_subset(rng_state, "ww_val", transform=trans)
    valwl_data = dataset.get_subset(rng_state, "wl_val", transform=trans)
    valll_data = dataset.get_subset(rng_state, "ll_val", transform=trans)
    vallw_data = dataset.get_subset(rng_state, "lw_val", transform=trans)
    testww_data = dataset.get_subset(rng_state, "ww_val", transform=trans)
    testwl_data = dataset.get_subset(rng_state, "wl_val", transform=trans)
    testll_data = dataset.get_subset(rng_state, "ll_val", transform=trans)
    testlw_data = dataset.get_subset(rng_state, "lw_val", transform=trans)

    ##### training data min group #####
    num_wl_points = num_minority_points //2
    num_lw_points = num_minority_points //2
    
    num_ww_points = num_majority_points //2
    num_ll_points = num_majority_points //2
    num_ww_points_per_group = num_ww_points //(n_maj_sources)
    num_ll_points_per_group = num_ll_points //(n_maj_sources)
    rng_state = np.random.get_state()
    trainingwl_data = dataset.get_subset(
        rng_state, "wl_train", num_points=num_wl_points, transform=trans, source_id=0)
    traininglw_data = dataset.get_subset(
        rng_state, "lw_train", num_points=num_lw_points, transform=trans, source_id=0)
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
        ww_data_one_group = dataset.get_subset(
            rng_state, "ww_train", sample_idx=indxs_to_sample_ww, source_id=i, transform=trans)
        ll_data_one_group = dataset.get_subset(
            rng_state, "ll_train", sample_idx=indxs_to_sample_ll, source_id=i, transform=trans)
        one_source = torch.utils.data.ConcatDataset([ww_data_one_group, ll_data_one_group])
        data_sources[i] = one_source
    val_data_dict = {'ww_val': valww_data, 'll_val': valll_data,
                      'lw_val': vallw_data, 'wl_val': valwl_data}
    test_data_dict = {'ww_test': testww_data, 'll_test': testll_data,
                      'lw_test': testlw_data, 'wl_test': testwl_data}
    return dataset, data_sources, val_data_dict, test_data_dict

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

    print(f"num male blond: {len(all_mb)}")
    print(f'num female blond: {len(all_fb)}')
    print(f'num male not blond : {len(all_mnb)}')
    print(f'num female not blond: {len(all_fnb)}')

    max_mb_points = len(all_mb)
    max_fb_points = len(all_fb)
    max_mnb_points = len(all_mnb)
    max_fnb_points = len(all_fnb)
    
    val_mb = CelebA(root_dir, download=True, transform=trans, split='val_mb')
    val_fb = CelebA(root_dir, download=True, transform=trans, split='val_fb')
    val_mnb = CelebA(root_dir, download=True, transform=trans, split='val_mnb')
    val_fnb = CelebA(root_dir, download=True, transform=trans, split='val_fnb')

    test_mb = CelebA(root_dir, download=True, transform=trans, split='test_mb')
    test_fb = CelebA(root_dir, download=True, transform=trans, split='test_fb')
    test_mnb = CelebA(root_dir, download=True, transform=trans, split='test_mnb')
    test_fnb = CelebA(root_dir, download=True, transform=trans, split='test_fnb')
    
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
                      'mnb_test': test_mnb, 'fnb_test': test_fnb}
    val_data_dict = {'mb_val': val_mb, 'fb_val': val_fb,
                      'mnb_val': val_mnb, 'fnb_val': val_fnb}

    return test_mb, data_sources, val_data_dict, test_data_dict


def iwildcam_n_sources(n_sources, max_training_data_size=None, img_size=None):
    if img_size == None:
        img_size = 512
    trans = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])


    dataset = IWildCamDataset(root_dir='/network/scratch/m/mizu.nishikawa-toomey')
    points = dataset.list_number_of_points_per_env()
    sorted_points = {k: v for k, v in sorted(points.items(), key=lambda item: item[1])}
    sorted_keys_top_15 = [*sorted_points.keys()][-15:]
    # [101, 255, 188, 120, 2, 296, 307, 221, 139, 26, 54, 265, 230, 187, 288]
    sorted_values_top_15 = [*sorted_points.values()][-15:]
    # [3020, 3176, 3441, 3499, 3520, 3550, 3559, 3722, 3766, 3960, 3990, 4010, 4439, 7600, 8494]
    data_sources = collections.defaultdict()
    data_sources_test = collections.defaultdict()
    for i in range(n_sources):
        testing = dataset.get_subset_based_on_metadata(
            sorted_keys_top_15[-i-1], index_col='location_remapped', sample_idx=[j for j in range(200)], transform=trans)
        if max_training_data_size == None:
            sample_idx = [j for j in range(200, sorted_values_top_15[-i-1])]
        else:
            sample_idx = [j for j in range(200, max_training_data_size + 200)]
        training = dataset.get_subset_based_on_metadata(sorted_keys_top_15[-i-1], index_col='location_remapped', sample_idx=sample_idx,
                                                        transform=trans)
        data_sources[sorted_keys_top_15[-i-1]] = training
        data_sources_test[sorted_keys_top_15[-i-1]] = testing
    return dataset, data_sources, data_sources_test

def camelyon17(max_training_data_size, source_proportions=[], img_size=None):
    if len(source_proportions) >0:
        assert math.isclose(sum(source_proportions),1, rel_tol=0.02)
    if img_size == None:
        img_size = 96
    trans = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    dataset = Camelyon17Dataset(root_dir='/tmp/')
    index_col = 0
    max_points = dataset.list_number_of_points_per_source()
    print(max_points)
    data_sources = collections.defaultdict()
    data_sources_test = collections.defaultdict()
    data_sources_val = collections.defaultdict()
    rng_state = np.random.get_state()
    for i in range(5):
        testing = dataset.get_subset_based_on_metadata(
            rng_state, i, index_col, sample_idx=[j for j in range(400)],
            transform=trans, source_id=i)
        sample_idx_val = [j for j in range(400, 800)]
        val = dataset.get_subset_based_on_metadata(rng_state, i, index_col, sample_idx=sample_idx_val,
                                                        transform=trans, source_id=i)
        if max_training_data_size == None:
            sample_idx_train = [j for j in range(800, max_points[i])]
        else:
            if len(source_proportions) == 0:
                data_size = max_training_data_size // 5
            else:
                data_size = int(max_training_data_size*source_proportions[i])
                sample_idx_train = [j for j in range(800, data_size)]
        training = dataset.get_subset_based_on_metadata(rng_state,
                                                        i, index_col, sample_idx=sample_idx_train,
                                                        transform=trans, source_id=i)
        data_sources[i] = training
        data_sources_test[i] = testing
        data_sources_val[i] = val
    return dataset, data_sources, data_sources_val, data_sources_test

def camelyon17_ood(max_training_data_size, group_proportions=[], test_source=0, img_size=None):
    if len(group_proportions) >0:
        assert math.isclose(sum(group_proportions),1, rel_tol=0.02)
    
    if img_size == None:
        img_size = 96
    trans = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    group_string_map_test = {test_source: test_source}
    dataset = Camelyon17Dataset(root_dir='/tmp/')
    index_col = 0
    max_points = dataset.list_number_of_points_per_source()
    print(max_points)
    data_sources = collections.defaultdict()
    data_sources_test = collections.defaultdict()
    rng_state = np.random.get_state()
    train_sources = [i for i in range(5)]
    train_sources.pop(test_source)
    for i in train_sources:
        if max_training_data_size == None:
            sample_idx_train = [j for j in range(400, max_points[i])]
        else:
            if len(group_proportions) == 0:
                data_size = max_training_data_size // 5
            else:
                data_size = int(max_training_data_size*group_proportions[i])
            sample_idx = [j for j in range(400, 400 + data_size)]
        training = dataset.get_subset_based_on_metadata(rng_state, i, index_col, sample_idx=sample_idx,
                                                        transform=trans, source_id=i)
        data_sources[i] = training
    data_sources_test[test_source] = dataset.get_subset_based_on_metadata(
        rng_state, test_source, index_col, sample_idx=[j for j in range(400)],
        transform=trans, source_id=test_source)

    return dataset, data_sources, data_sources_test

def fmow(max_training_data_size, group_proportions=[], img_size=None, num_sources=5, test_size=400):
    if len(group_proportions) >0:
        assert math.isclose(sum(group_proportions),1, rel_tol=0.02), f"group proportions dont sum to one, sums to {sum(group_proportions)}"
    
    if img_size == None:
        img_size = 224
    trans = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    
    dataset = FMoWDataset(root_dir='/network/scratch/m/mizu.nishikawa-toomey')
    index_col = 0
    max_points = dataset.list_number_of_points_per_source()
    print(f"number of points in each source in whole dataset {max_points}")
    data_sources = collections.defaultdict()
    data_sources_test = collections.defaultdict()
    rng_state = np.random.get_state()
    for i in range(num_sources):
        testing = dataset.get_subset_based_on_metadata(rng_state, i, index_col, sample_idx=[j for j in range(test_size)], transform=trans, source_id=i)
        if max_training_data_size == None:
            sample_idx_train = [j for j in range(test_size, max_points[i])]
            data_size = len(sample_idx_train)
        else:
            if len(group_proportions) == 0:
                data_size = max_training_data_size // 5
            else:
                data_size = int(max_training_data_size*group_proportions[i])
            sample_idx_train = [j for j in range(test_size, data_size + test_size)]
        assert max_points[i] >= data_size + test_size, f"for source {i}, requested points {data_size + test_size} is larger than availabe data size {max_points[i]}"

        training = dataset.get_subset_based_on_metadata(rng_state, i, index_col, sample_idx=sample_idx_train,
                                                        transform=trans, source_id=i)
        data_sources[i] = training
        data_sources_test[i] = testing
    return dataset, data_sources, data_sources_test

def fmow_ood(max_training_data_size, group_proportions=[], img_size=None, num_sources=5, test_size=400, test_source=0):
    if len(group_proportions) >0:
        assert math.isclose(sum(group_proportions),1, rel_tol=0.02), f"group proportions dont sum to one, sums to {sum(group_proportions)}"
    
    if img_size == None:
        img_size = 224
    trans = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    
    dataset = FMoWDataset(root_dir='/tmp/')
    index_col = 0
    max_points = dataset.list_number_of_points_per_source()
    print(f"number of points in each source in whole dataset {max_points}")
    data_sources = collections.defaultdict()
    data_sources_test = collections.defaultdict()
    rng_state = np.random.get_state()
    testing = dataset.get_subset_based_on_metadata(rng_state, test_source, index_col, sample_idx=[j for j in range(test_size)], transform=trans, source_id=test_source)
    train_sources = [i for i in range(5)]
    train_sources.pop(test_source)

    data_sources_test[test_source] = testing
    for i in train_sources:
        if max_training_data_size == None:
            sample_idx_train = [j for j in range(test_size, max_points[i])]
            data_size = len(sample_idx_train)
        else:
            if len(group_proportions) == 0:
                data_size = max_training_data_size // 5
            else:
                data_size = int(max_training_data_size*group_proportions[i])
            sample_idx_train = [j for j in range(test_size, data_size + test_size)]
        assert max_points[i] >= data_size + test_size, f"for source {i}, requested points {data_size + test_size} is larger than availabe data size {max_points[i]}"

        training = dataset.get_subset_based_on_metadata(rng_state, i, index_col, sample_idx=sample_idx_train,
                                                        transform=trans, source_id=i)
        data_sources[i] = training
    return dataset, data_sources, data_sources_test

def cmnist_n_sources(num_minority_points, num_majority_points,
                     n_maj_sources, causal_noise=0, spurious_noise=0, num_digits_per_target=5, binary_classification=True):
    trans = transforms.Compose([transforms.ToTensor()])
    start_idx = 0
    data_sources = collections.defaultdict()
    if num_digits_per_target == 1:
        multiplier = 5
    else:
        multiplier = 1
    dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                              causal_noise=causal_noise,
                              transform=trans, start_idx=start_idx, num_samples=num_minority_points*multiplier, 
                              red=1, source_id=0, num_digits_per_target=num_digits_per_target,
                              binary_classification=binary_classification, train=True)
    data_sources[0] = dataset
    start_idx += num_minority_points*multiplier

    num_majority_points_per_group = num_majority_points // n_maj_sources
    for i in range(1, n_maj_sources + 1):
        dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                  causal_noise=causal_noise,
                                  transform=trans, start_idx=start_idx, num_samples=num_majority_points_per_group*multiplier, 
                                  red=0, source_id=i, num_digits_per_target=num_digits_per_target,
                                  binary_classification=binary_classification, train=True)
        start_idx += num_majority_points_per_group*multiplier
        data_sources[i] = dataset

    datasety0r_val = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                        causal_noise=0,
                                        transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                        source_id=0, red=0, specified_class = 0,
                                        num_digits_per_target=num_digits_per_target,binary_classification=binary_classification, train=True)
    datasety1g_val = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                        causal_noise=0,
                                        transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                        source_id=0, red=0, specified_class =1,
                                        num_digits_per_target=num_digits_per_target,binary_classification=binary_classification, train=True)

    start_idx += 5000
    datasety0g_val = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                      causal_noise=0, specified_class=0,
                                      transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                      source_id=1, red=1, num_digits_per_target=num_digits_per_target,
                                      binary_classification=binary_classification,train=True)

    datasety1r_val = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                        causal_noise=0, specified_class=1,
                                        transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                        source_id=1, red=1,num_digits_per_target=num_digits_per_target,
                                        binary_classification=binary_classification,train=True)

    start_idx = 0
    datasety0r_test = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                        causal_noise=0,
                                        transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                        source_id=0, red=0, specified_class = 0,
                                        num_digits_per_target=num_digits_per_target,binary_classification=binary_classification,train=False)
    datasety1g_test = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                        causal_noise=0,
                                        transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                        source_id=0, red=0, specified_class =1,
                                        num_digits_per_target=num_digits_per_target,binary_classification=binary_classification, train=False)

    start_idx += 5000
    datasety0g_test = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                      causal_noise=0, specified_class=0,
                                      transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                      source_id=1, red=1,num_digits_per_target=num_digits_per_target,
                                      binary_classification=binary_classification,train=False)

    datasety1r_test = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                        causal_noise=0, specified_class=1,
                                        transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                        source_id=1, red=1,num_digits_per_target=num_digits_per_target,
                                        binary_classification=binary_classification,train=False)

    data_sources_val = {'y0r_val': datasety0r_val, 'y1r_val': datasety1r_val,
                        'y0g_val': datasety0g_val, 'y1g_val': datasety1g_val}

    data_sources_test = {'y0r_test': datasety0r_test, 'y1r_test': datasety1r_test,
                         'y0g_test': datasety0g_test, 'y1g_test': datasety1g_test}

    return dataset, data_sources, data_sources_test, data_sources_val

def cmnist_10_n_sources(num_minority_points, num_majority_points,
                        n_maj_sources, causal_noise=0, spurious_noise=0, num_digits_per_target=5, binary_classification=False):
    multiplier = 1
    trans = transforms.Compose([transforms.ToTensor()])
    start_idx = 0
    data_sources = collections.defaultdict()
    dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                              causal_noise=causal_noise,
                              transform=trans, start_idx=start_idx, num_samples=num_minority_points*multiplier, 
                              red=1, source_id=0, num_digits_per_target=num_digits_per_target,
                              binary_classification=binary_classification, train=True)
    data_sources[0] = dataset
    start_idx += num_minority_points*multiplier

    num_majority_points_per_group = num_majority_points // n_maj_sources
    for i in range(1, n_maj_sources + 1):
        dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                  causal_noise=causal_noise,
                                  transform=trans, start_idx=start_idx, num_samples=num_majority_points_per_group*multiplier, 
                                  red=0, source_id=i, num_digits_per_target=num_digits_per_target,
                                  binary_classification=binary_classification, train=True)
        start_idx += num_majority_points_per_group*multiplier
        data_sources[i] = dataset
        
    val_dict = collections.defaultdict()
    test_dict = collections.defaultdict()
    for i in range(10):
        datasetyir_val = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                         causal_noise=0,
                                         transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                         source_id=0, red=1, specified_class = i,
                                         num_digits_per_target=1,
                                         binary_classification=False, train=True)
        val_dict[f'{i}_r'] = datasetyir_val

    start_idx += 5000
    for i in range(10):
        datasetyig_val = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                         causal_noise=0,
                                         transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                         source_id=0, red=0, specified_class = i,
                                         num_digits_per_target=1,
                                         binary_classification=False,  train=True)
        val_dict[f'{i}_g'] = datasetyig_val
    start_idx = 0
    for i in range(10):
        datasetyir_test = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                          causal_noise=0,
                                          transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                          source_id=0, red=1, specified_class = i,
                                          num_digits_per_target=1,
                                          binary_classification=False, train=False)
        test_dict[f'{i}_r'] = datasetyir_test
    start_idx += 5000
    for i in range(10):
        datasetyig_test = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                          causal_noise=0,
                                          transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                          source_id=0, red=0, specified_class = i,
                                          num_digits_per_target=1,
                                          binary_classification=False,  train=False)
        test_dict[f'{i}_g'] =  datasetyig_test

    return dataset, data_sources, test_dict, val_dict

def cmnist_n_sources_ood(num_minority_points, num_majority_points,
                     n_maj_sources, causal_noise=0, spurious_noise=0, num_digits_per_target=5, binary_classification=True):
    trans = transforms.Compose([transforms.ToTensor()])
    start_idx = 0
    data_sources = collections.defaultdict()
    if num_digits_per_target == 1:
        multiplier = 5
    else:
        multiplier = 1
    dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                              causal_noise=spurious_noise,
                              transform=trans, start_idx=start_idx, num_samples=num_minority_points*multiplier, 
                              red=0, source_id=0, num_digits_per_target=num_digits_per_target,
                              binary_classification=binary_classification)
    
    test_set = {'y0r_y1g': dataset}
    start_idx += num_minority_points*multiplier

    num_majority_points_per_group = num_majority_points // n_maj_sources
    for i in range(0, n_maj_sources):
        dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                  causal_noise=causal_noise,
                                  transform=trans, start_idx=start_idx, num_samples=num_majority_points_per_group*multiplier, 
                                  red=1, source_id=i,num_digits_per_target=num_digits_per_target,
                                  binary_classification=binary_classification)
        start_idx += num_majority_points_per_group*multiplier
        data_sources[i] = dataset
    datasety0r_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                        causal_noise=0,
                                        transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                        source_id=0, red=0, specified_class = 0,
                                        num_digits_per_target=num_digits_per_target,binary_classification=binary_classification)
    datasety1g_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                        causal_noise=0,
                                        transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                        source_id=0, red=0, specified_class =1,
                                        num_digits_per_target=num_digits_per_target,binary_classification=binary_classification)
    start_idx += 5000
    datasety0g_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                      causal_noise=0, specified_class=0,
                                      transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                      source_id=1, red=1,num_digits_per_target=num_digits_per_target,
                                      binary_classification=binary_classification)

    datasety1r_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                        causal_noise=0, specified_class=1,
                                        transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                        source_id=1, red=1,num_digits_per_target=num_digits_per_target,
                                        binary_classification=binary_classification)

    return dataset, data_sources, {'y0r': datasety0r_unseen, 'y1r': datasety1r_unseen,
                                   'y0g': datasety0g_unseen, 'y1g': datasety1g_unseen}


def cmnist_n_sources_diff_env(num_minority_points, num_majority_points,
                     n_maj_sources, causal_noise=0, spurious_noise=0, num_digits_per_target=5):
    trans = transforms.Compose([transforms.ToTensor()])
    start_idx = 0
    data_sources = collections.defaultdict()
    if num_digits_per_target == 1:
        multiplier = 5
    else:
        multiplier = 1
    dataset = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                              causal_noise=0,
                              transform=trans, start_idx=start_idx, num_samples=num_minority_points, 
                              red=0, source_id=0, num_digits_per_target=5)
    data_sources[0] = dataset
    start_idx += num_minority_points

    num_majority_points_per_group = num_majority_points // n_maj_sources
    for i in range(1, n_maj_sources + 1):
        dataset = ColoredMNISTRAM(root='./data', spurious_noise=spurious_noise, 
                                  causal_noise=causal_noise,
                                  transform=trans, start_idx=start_idx, num_samples=num_majority_points_per_group*multiplier, 
                                  red=1, source_id=i,num_digits_per_target=num_digits_per_target)
        start_idx += num_majority_points_per_group*multiplier
        data_sources[i] = dataset
    dataset0_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                      causal_noise=0,
                                      transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                      source_id=0, red=0,num_digits_per_target=num_digits_per_target)
    start_idx += 5000
    dataset1_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                      causal_noise=0,
                                      transform=trans, start_idx=start_idx, num_samples=5000*multiplier,
                                      source_id=1, red=1,num_digits_per_target=num_digits_per_target)

        
    return dataset, data_sources, {'y0r': dataset0_unseen, 'y1r': dataset1_unseen}


if __name__ == "__main__":
    # Load data from the Camelyon17 dataset
    dataset, data_sources, data_sources_test = camelyon17(max_training_data_size=2000)
    
    # Count the number of classes in each source
    print("\n=== Number of classes per source ===")
    for source_id, source_data in data_sources.items():
        # Extract all labels from the source
        labels = []
        for item in source_data:
            label = item['target']
            labels.append(label.item() if hasattr(label, 'item') else label)
        # Count unique classes
        unique_classes = set(labels)
        num_classes = len(unique_classes)
        
        print(f"Source {source_id}: {num_classes} classes - Classes: {sorted(unique_classes)}")
        print(f"  Total samples: {len(labels)}")
        print(f"  Class distribution: {collections.Counter(labels)}")
    
    # from keras.utils import HDF5Matrix
    
    # from keras.preprocessing.image import ImageDataGenerator

    # x_train = HDF5Matrix('camelyonpatch_level_2_split_train_x.h5-002', 'x')l
    # y_train = HDF5Matrix('camelyonpatch_level_2_split_train_y.h5', 'y')
