from torchvision import transforms
import pickle
import numpy as np
import collections
import sys
import torch
import models
#from models import BayesianNet, resnet50, BayesianNetRes50, BayesianNetFc, Linear, ConvNet, resnet50_all
from active_learning_data import ActiveLearningDataGroups
from tools import calc_ent_batched, calc_ent_per_group_batched, plot_dictionary, log_dict
from pprint import pprint
import wandb
from torch.utils.data import DataLoader, ConcatDataset, Subset
import random
from tools import slurm_infos
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from waterbirds_dataset import WaterbirdsDataset, train_batched, test_batched, test_per_group
from acquisitions import (Random, UniformGroups,
                          EntropyPerGroup, AccuracyPerGroup, Entropy, EntropyUniformGroups)

# to turn off wandb, export WANDB_MODE=disabled
def main(args):
    wandb.init(
        project=args.project_name,
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    wandb.run.summary.update(slurm_infos())

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = True
    #log = {'train_acc':[], 'ent1': [], 'cross_ent_1': [],
    #       'ent2': [], 'cross_ent_2': [], 'test_acc':[],
    #       'num points': [], 'causal acc': [], 'sp acc': []}
    log = collections.defaultdict(list)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.use_rep:
        trans = None
        file_name = 'data/waterbirds_v1.0/waterbirds_resnet50/normalisation.pkl'
        with open(file_name, 'rb') as f:
            norm_dict = pickle.load(f)
    else:
        trans = transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        )
        norm_dict = None
    # training datasets
    split_scheme = {"g0_train":0, "g1_train": 1,"g2_train": 2, "g3_train": 3,
                    "ww_test":4, "wl_test": 5,"ll_test": 6, "lw_test": 7, 'test': 8}
    split_names = {"g0_train":'g0_train', "g1_train": 'g1_train', "g2_train": 'g2_train',
                   "g3_train": 'g3_train', "ww_test":"ww_test", "wl_test": "wl_test","ll_test": "ll_test",
                   "lw_test": "lw_test", "test": "test"}

    dataset = WaterbirdsDataset(version='1.0', root_dir='data/', download=True,
                                split_scheme=split_scheme, split_names=split_names,
                                metadata_name=args.data_mode, use_rep=args.use_rep)
    training0_data = dataset.get_subset(
        "g0_train", transform=trans)
    training1_data = dataset.get_subset(
        "g1_train",
        transform=trans)
    training2_data = dataset.get_subset(
        "g2_train",
        transform=trans)
    training3_data = dataset.get_subset(
        "g3_train",
        transform=trans)
    testww_data = torch.utils.data.DataLoader(
        dataset.get_subset(
            "ww_test",
            transform=trans), batch_size=args.batch_size, **kwargs)

    testwl_data = torch.utils.data.DataLoader(
        dataset.get_subset(
            "wl_test",
            transform=trans), batch_size=args.batch_size, **kwargs)
    testll_data = torch.utils.data.DataLoader(
        dataset.get_subset(
        "ll_test", transform=trans), batch_size=args.batch_size, **kwargs)
    testlw_data = torch.utils.data.DataLoader(
        dataset.get_subset(
        "lw_test",
        transform=trans), batch_size=args.batch_size, **kwargs)
    dataset_test = dataset.get_subset(
        "test",
        transform=trans)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, **kwargs)
    group_to_log1 = 0
    group_to_log2 = 3
    model = getattr(models, args.model_name)
    if args.balanced:
        data_train = [training0_data, training1_data]
    elif args.maj_group_only:
        data_train = [training1_data, training2_data, training3_data]
    else:
        data_train = [training0_data, training1_data, training2_data, training3_data]
        
    num_groups = len(data_train)
    samples_per_group = int(args.al_size / num_groups)
    if args.train_all_data:
        model = model(2)
        dataset = ConcatDataset(data_train)
        if args.size != -1:
            list_idx = [i for i in range(len(dataset))]
            items = random.sample(list_idx, args.size)
            dataset = Subset(dataset, items)
        dataloader_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        proportion_correct_train, proportion_correct_test, group_dict_train, wga = train_batched(
                    model=model, dataloader=dataloader_train,
            dataloader_test=dataloader_test, lr=args.lr, num_epochs=args.num_epochs,
            num_groups=num_groups, wandb=wandb, gdro=args.gdro, weight_decay=args.weight_decay)
        wandb.log({'num points': len(dataset)})
        sys.exit()
    group_dict = {key: samples_per_group for key in range(num_groups)}

    al_data = ActiveLearningDataGroups(data_train, dataset_test, 2, args.batch_size)
    method_map = {
        'random': Random,
        'uniform_groups': UniformGroups,
        'entropy_per_group': EntropyPerGroup,
        'accuracy': AccuracyPerGroup,
        'entropy': Entropy,
        'entropy_uniform_groups': EntropyUniformGroups}

    kwargs_map = {'random': {'al_data': al_data, 'al_size': args.al_size},
                  'uniform_groups': {'al_data': al_data, 'group_proportions': group_dict},
                  'entropy_per_group': {'al_data': al_data, 'al_size':args.al_size},
                  'entropy': {'al_data': al_data, 'al_size': args.al_size},
                  'accuracy': {'al_data': al_data, 'al_size': args.al_size},
                  'entropy_uniform_groups':{'al_data': al_data, 'al_size': args.al_size,
                                            'group_proportions': group_dict}}
    # initial random or uniform acquisition to start with
    acquisition_method = method_map[args.start_acquisition](**kwargs_map[args.start_acquisition])
    indices = acquisition_method.return_indices()
    al_data.acquire_with_indices(indices)
        
    for i in range(1, args.al_iters):
        print('al iteration: ', i)
        # setting up trainig
        acquisition_method = method_map[args.acquisition](**kwargs_map[args.acquisition])
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = getattr(models, args.model_name)
        model = model(2)
        dataloader_train, dataloader_test = al_data.get_train_and_test_loader(batch_size=args.batch_size)
        num_points = len(al_data.train.indices)
        proportion_correct_train, proportion_correct_test, group_dict_train, wga = train_batched(
            model=model, dataloader=dataloader_train,
            dataloader_test=dataloader_test, lr=args.lr, num_epochs=args.num_epochs,
            num_groups=num_groups, norm_dict=norm_dict, weight_decay=args.weight_decay)
        # using model get info for acquisition function
        print('dict groups in train', group_dict_train)
        if args.acquisition in ['random', 'uniform_groups']:
            pass
        elif args.acquisition == 'entropy_per_group':
            acquisition_method.information_for_acquisition(model, num_groups)
        elif args.acquisition == 'entropy':
            acquisition_method.information_for_acquisition(model)
        elif args.acquisition == 'accuracy':
            acquisition_method.information_for_acquisition(model, indices, num_groups, k=3)
        elif args.acquisition == 'entropy_uniform_groups':
            acquisition_method.information_for_acquisition(model, num_groups)
        else:
            print('acquisition not recognised')

        # acquire data
        indices = acquisition_method.return_indices()
        al_data.acquire_with_indices(indices)
        # compute metrics and logging
        entww, _ = calc_ent_batched(model, testww_data, num_models=100)
        entwl, _ = calc_ent_batched(model, testwl_data, num_models=100)
        entll, _ = calc_ent_batched(model, testll_data, num_models=100)
        entlw, _ = calc_ent_batched(model, testlw_data, num_models=100)
        to_log = {'train_acc': proportion_correct_train,
                  'num points':num_points,
                  'g0 points': group_dict_train[group_to_log1],
                  'g1 points': group_dict_train[group_to_log2],
                  'entww': entww,
                  'entwl': entwl,
                  'entll': entll,
                  'entlw': entlw,
                  'wga': wga}
        to_log.update({'g' + str(key) + ' points' : value for key, value in group_dict_train.items()})
        if isinstance(proportion_correct_test, dict):
            to_log.update(proportion_correct_test)
        else:
            to_log.update({'test acc': proportion_correct_test})
        wandb.log(to_log) 
        pprint(to_log)
        log = log_dict(log, to_log)
    plot_dictionary(log)
    return log

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--al_iters', type=int, default=15)
    parser.add_argument('--al_size', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--size', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--acquisition', type=str, default='random')
    parser.add_argument('--model_name', type=str, default='BayesianNetRes50U')
    parser.add_argument('--start_acquisition', type=str, default='random')
    parser.add_argument('--project_name', type=str, default='al_wg')
    parser.add_argument('--data_mode', type=str, default='metadata_v7.csv')
    parser.add_argument('--use_rep', default=False, action='store_true')
    parser.add_argument('--gdro', default=False, action='store_true')
    parser.add_argument('--train_all_data', default=False, action='store_true')
    parser.add_argument('--balanced', default=False, action='store_true')
    parser.add_argument('--maj_group_only', default=False, action='store_true')
    
    args = parser.parse_args()

    main(args)
