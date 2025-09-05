from torchvision import transforms
import numpy as np
import collections
import torch
from models import BayesianNet, resnet50, BayesianNetRes50
from active_learning_data import ActiveLearningDataGroups
from tools import calc_ent_batched, calc_ent_per_group_batched, plot_dictionary, log_dict
from pprint import pprint
from acquisitions import Random, UniformGroups, EntropyPerGroup, AccuracyPerGroup, Entropy
import wandb
from tools import slurm_infos
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from waterbirds_dataset import WaterbirdsDataset, train_batched, test_batched, test_per_group


# to turn off wandb, export WANDB_MODE=disabled
def main(seed, project_name='al_wg_test', al_iters=10, al_size=100, num_epochs=150,
         acquisition='random', data1_size=5000,
         data2_size=1000, start_acquisition='uniform_groups', data_mode='two_groups'):
    wandb.init(
        project=project_name,
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    wandb.run.summary.update(slurm_infos())

    np.random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = True
    #log = {'train_acc':[], 'ent1': [], 'cross_ent_1': [],
    #       'ent2': [], 'cross_ent_2': [], 'test_acc':[],
    #       'num points': [], 'causal acc': [], 'sp acc': []}
    log = collections.defaultdict(list)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    # training datasets
    split_scheme = {"g0_train":10, "g1_train": 11,"g2_train": 12, "g3_train": 13,
    "g0_test":20, "g1_test": 21,"g2_test": 22, "g3_test": 23, 'train':0, 'val':1, 'test':2}
    split_names = {"g0_train":'g0_train', "g1_train": 'g1_train', "g2_train": 'g2_train',
                   "g3_train": 'g3_train', "g0_test": 'g0_test',
                   "g1_test": 'g1_test',"g2_test": 'g2_test',
                   "g3_test": 'g3_test', 'train':'train', 'test':'test', 'val':'val'}

    dataset = WaterbirdsDataset(version='1.0', root_dir='data/', download=True,
                          split_scheme=split_scheme, split_names=split_names,
                                metadata_name='metadata_5g_v2.csv')
    training0_data = dataset.get_subset(
        "g0_train",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    training1_data = dataset.get_subset(
        "g1_train",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    training2_data = dataset.get_subset(
        "g2_train",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    training3_data = dataset.get_subset(
        "g3_train",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )

    test0_data = dataset.get_subset(
        "g0_test",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    test1_data = dataset.get_subset(
        "g1_test",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    test2_data = dataset.get_subset(
        "g2_test",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    test3_data = dataset.get_subset(
        "g3_test",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    dataset_val = dataset.get_subset(
        "val",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    group_to_log1 = 0
    group_to_log2 = 3
    training0_data.pseudo_group_label = 0
    training1_data.pseudo_group_label = 1
    training2_data.pseudo_group_label = 2
    training3_data.pseudo_group_label = 3

    data_train = [training0_data, training1_data, training2_data, training3_data]
    num_groups = len(data_train)
    samples_per_group = int(al_size / num_groups)
    
    group_dict = {key: samples_per_group for key in range(num_groups)}
    dataset1_unseen = test0_data
    dataset2_unseen = test3_data

    dataloader1_unseen = torch.utils.data.DataLoader(dataset1_unseen,
                                                      batch_size=64, **kwargs)
    dataloader2_unseen = torch.utils.data.DataLoader(dataset2_unseen,
                                                      batch_size=64, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=64, **kwargs)

    al_data = ActiveLearningDataGroups(data_train, dataset_val, 2)
    method_map = {
        'random': Random,
        'uniform_groups': UniformGroups,
        'entropy_per_group': EntropyPerGroup,
        'accuracy': AccuracyPerGroup,
    'entropy': Entropy}

    kwargs_map = {'random': {'al_data': al_data, 'al_size': al_size},
                  'uniform_groups': {'al_data': al_data, 'group_proportions': group_dict},
                  'entropy_per_group': {'al_data': al_data, 'al_size': al_size},
                  'entropy': {'al_data': al_data, 'al_size': al_size},
                  'accuracy': {'al_data': al_data, 'al_size': al_size}}
    # initial random or uniform acquisition to start with
    acquisition_method = method_map[start_acquisition](**kwargs_map[start_acquisition])
    indices = acquisition_method.return_indices()
    al_data.acquire_with_indices(indices)

    if acquisition == 'accuracy':
        # first we have training on random acquisiion
        # when using cross entropy based acquisition, we require 
        # two uniform group acquisitions
        model = resnet50(classes=2)
        dataloader_train, dataloader_test = al_data.get_train_and_test_loader(batch_size=64)
        # train model1 on first batch D1
        proportion_correct_train, proportion_correct_test, group_dict_train = train_batched(
            model=model, dataloader=dataloader_train,
            dataloader_test=dataloader_test, lr=0.001, epochs=num_epochs, num_groups=num_groups)
        ent1, cross_ent1 = calc_ent_batched(model, dataloader1_unseen, num_models=100)
        ent2, cross_ent2 = calc_ent_batched(model, dataloader2_unseen, num_models=100)
        num_points = len(al_data.train.indices)
        to_log = {'train_acc': proportion_correct_train, 'test_acc': proportion_correct_test,
                  'cross_ent_1': cross_ent1, 'cross_ent_2': cross_ent2,
                  'num points':num_points, 'ent1': ent1, 'ent2' :ent2,
                  'g0 points': group_dict_train[group_to_log1],
                  'g1 points': group_dict_train[group_to_log2]}
        wandb.log(to_log)
        # second random acquisition
        
        acquisition_method = method_map[acquisition](**kwargs_map[acquisition])
        # sample uniformly from groups again to get D2
        indices2 = acquisition_method.return_indices_random()
        al_data.acquire_with_indices(indices2)
        # test D2 on model 1 to obtain group sampling amounts 
        
        # third, non-random acquisition
        acquisition_method.information_for_acquisition(model, indices2, num_groups, k=3)
        # decide on the next aqcuisition based on the test accuracy results. 
        indices = acquisition_method.return_indices()
        al_data.acquire_with_indices(indices)
        
    for i in range(1, al_iters):
        print('al iteration: ', i)
        # setting up trainig
        num_epochs += 50
        acquisition_method = method_map[acquisition](**kwargs_map[acquisition])
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = BayesianNetRes50(num_classes=2)
        dataloader_train, dataloader_test = al_data.get_train_and_test_loader(batch_size=64)
        num_points = len(al_data.train.indices)
        proportion_correct_train, proportion_correct_test, group_dict_train = train_batched(
            model=model, dataloader=dataloader_train,
            dataloader_test=dataloader_test, lr=0.001, epochs=num_epochs, num_groups=num_groups)
        # using model get info for acquisition function
        print('dict groups in train', group_dict_train)
        if acquisition in ['random', 'uniform_groups']:
            pass
        elif acquisition == 'entropy_per_group':
            acquisition_method.information_for_acquisition(model, num_groups)
        elif acquisition == 'entropy':
            acquisition_method.information_for_acquisition(model)
        elif acquisition == 'accuracy':
            acquisition_method.information_for_acquisition(model, indices, num_groups, k=3)
        else:
            print('acquisition not recognised')
        # acquire data
        indices = acquisition_method.return_indices()
        al_data.acquire_with_indices(indices)
        # compute metrics and logging
        ent1, cross_ent1 = calc_ent_batched(model, dataloader1_unseen, num_models=100)
        ent2, cross_ent2 = calc_ent_batched(model, dataloader2_unseen, num_models=100)
        to_log = {'train_acc': proportion_correct_train, 'test_acc': proportion_correct_test,
                  'cross_ent_1': cross_ent1, 'cross_ent_2': cross_ent2,
                  'num points':num_points, 'ent1': ent1, 'ent2' :ent2,
                  'g0 points': group_dict_train[group_to_log1],
                  'g1 points': group_dict_train[group_to_log2]}
        wandb.log(to_log)
        pprint(to_log)
        log = log_dict(log, to_log)
    plot_dictionary(log)
    return log

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--al_iters', type=int, default=10)
    parser.add_argument('--al_size', type=int, default=100)
    parser.add_argument('--data1_size', type=int, default=5000)
    parser.add_argument('--data2_size', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--acquisition', type=str, default='random')
    parser.add_argument('--start_acquisition', type=str, default='random')
    parser.add_argument('--project_name', type=str, default='al_wg')
    parser.add_argument('--data_mode', type=str, default='two_groups')
    
    args = parser.parse_args()

    main(**vars(args))
    seed = 0
