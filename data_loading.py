import torch
from torchvision import transforms
import random
from waterbirds_dataset import WaterbirdsDataset
from torchvision import transforms
from torchvision.datasets import CelebA
def waterbirds(num_minority_points, num_majority_points, batch_size,
               metadata_path='metadata_v8.csv', root_dir='data/'):
    use_cuda = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trans = transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    )
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
    testww_data = torch.utils.data.DataLoader(
        dataset.get_subset(
            "ww_test",
            transform=trans), batch_size=batch_size, **kwargs)
    testwl_data = torch.utils.data.DataLoader(
        dataset.get_subset(
            "wl_test",
            transform=trans), batch_size=batch_size, **kwargs)
    testll_data = torch.utils.data.DataLoader(
        dataset.get_subset(
        "ll_test", transform=trans), batch_size=batch_size, **kwargs)
    testlw_data = torch.utils.data.DataLoader(
        dataset.get_subset(
        "lw_test",
        transform=trans), batch_size=batch_size, **kwargs)
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
    test_data = {'ww_test': testww_data, 'll_test': testll_data,
                 'lw_test': testlw_data, 'wl_test': testwl_data, 'val': val_data}
    return training_data, test_data, training_data_dict

def celebA():
    trans = transforms.Compose([transforms.PILToTensor()])
    dataset = CelebA('/network/scratch/m/mizu.nishikawa-toomey', download=True, transform=trans)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for batch in dataloader:
        img, attributes = batch
        import pdb
        pdb.set_trace()
if __name__ == "__main__":

    celebA()
