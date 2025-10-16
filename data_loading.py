import torch
from torchvision import transforms
import random
from waterbirds_dataset import WaterbirdsDataset
from torchvision import transforms
from celeba import CelebA

def waterbirds(num_minority_points, num_majority_points, batch_size,
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
    training0_data = dataset.get_subset(
        "wl_train", transform=trans)
    training1_data = dataset.get_subset(
        "lw_train",
        transform=trans)
    training2_data = dataset.get_subset(
        "ww_train",
        transform=trans)
    training3_data = dataset.get_subset(
        "ll_train",
        transform=trans)
    training_data_dict = {'wl_train': training0_data, 'lw_train': training1_data,
                          'ww_train': training2_data, 'll_train': training3_data}

    ww_test = dataset.get_subset("ww_test", transform=trans)
    testww_data = torch.utils.data.DataLoader(ww_test, batch_size=batch_size, **kwargs)
    wl_test = dataset.get_subset("wl_test", transform=trans)
    testwl_data = torch.utils.data.DataLoader(wl_test, batch_size=batch_size,**kwargs)
    ll_test = dataset.get_subset("ll_test", transform=trans)
    testll_data = torch.utils.data.DataLoader(ll_test, batch_size=batch_size, **kwargs)
    lw_test = dataset.get_subset("lw_test", transform=trans)
    testlw_data = torch.utils.data.DataLoader(lw_test, batch_size=batch_size, **kwargs)
    val_data = torch.utils.data.DataLoader(
        dataset.get_subset(
        "val",
        transform=trans), batch_size=batch_size, **kwargs)
    print(f"Training data max sizes: wl: {len(training0_data)}, lw: {len(training1_data)}, ww: {len(training2_data)}, ll {len(training3_data)}")
    num_wl_points = num_minority_points //2
    num_lw_points = num_minority_points //2
    num_ww_points = num_majority_points //2
    num_ll_points = num_majority_points //2
    assert num_wl_points <= len(training0_data)
    assert num_lw_points <= len(training1_data)
    assert num_ww_points <= len(training2_data)
    assert num_ll_points <= len(training3_data)
    print(f"Training data used sizes wl : {num_wl_points}, lw : {num_lw_points}, ww: {num_ww_points}, ll: {num_ll_points}")
    idx_wl_points = random.sample([i for i in range(len(training0_data))], k=num_wl_points)
    idx_lw_points = random.sample([i for i in range(len(training1_data))], k=num_lw_points)
    idx_ww_points = random.sample([i for i in range(len(training2_data))], k=num_ww_points)
    idx_ll_points = random.sample([i for i in range(len(training3_data))], k=num_ll_points)
    data0 = torch.utils.data.Subset(training0_data, idx_wl_points)
    data1 = torch.utils.data.Subset(training1_data, idx_lw_points)
    data2 = torch.utils.data.Subset(training2_data, idx_ww_points)
    data3 = torch.utils.data.Subset(training3_data, idx_ll_points)
    training_data = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([data0, data1, data2, data3]), shuffle=True, batch_size=batch_size)
    test_data_dict = {'ww_test': testww_data, 'll_test': testll_data,
                      'lw_test': testlw_data, 'wl_test': testwl_data, 'val': val_data}
    test_data = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([ww_test, wl_test, ll_test, lw_test]), shuffle=True, batch_size=batch_size)
    return training_data, test_data, training_data_dict, test_data_dict


def celeba_load(num_minority_points, num_majority_points, batch_size, root_dir='/tmp/', img_size=None):
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
    training_data = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([data0, data1, data2, data3]), shuffle=True, batch_size=batch_size)
    test_data_dict = {'mb_test': blond_male_test, 'fb_test': blond_female_test,
                      'mnb_test': notblond_male_test, 'fnb_test': notblond_female_test, 'val': val}
    test_data = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([blond_male_test, blond_female_test, notblond_male_test, notblond_female_test]),
        shuffle=True, batch_size=batch_size)

    return training_data, test_data, training_data_dict, test_data_dict

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

if __name__ == "__main__":
    training_data, test_data, training_data_dict = celebA(100, 1000, 20)
