import collections
import itertools
import torch
import random
import wandb
import numpy as np
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from waterbirds_dataset import WaterbirdsDataset, train_batched, test_batched, test_per_group
from tools import provide_groupings_test_acc, provide_group_weights, slurm_infos
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from models import ConvNet, resnet50, wideresnet50, resnet50_plus

def calc_data_len_upsample(weights, data_list):
    smallest_weight = sorted(weights.values())[0]
    normalised_weights = {k: v/smallest_weight for k,v in weights.items()}
    normalised_weights_sorted = sorted(normalised_weights.items())
    data_size_sampled = [
        len(data) * norm_weight[1] for data, norm_weight in zip(data_list, normalised_weights_sorted)]
    total_data_size = sum(data_size_sampled)
    return int(total_data_size)

def main(num_epochs, lr, weight_decay, data_dir='data/',
         batch_size=128, debug=False, wandb_project='new_gdro', model='convnet', seed=0):
    wandb.init(
        project=wandb_project,
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    wandb.run.summary.update(slurm_infos())

    print('lr: ', lr)
    print('num_epochs: ', num_epochs)
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    if debug == True:
        num_epochs = 1
    torch.manual_seed(seed)
    split_scheme = {"g0_train":10, "g1_train": 11,"g2_train": 12, "g3_train": 13,
    "g0_test":20, "g1_test": 21,"g2_test": 22, "g3_test": 23, 'train':0, 'val':1, 'test':2}
    split_names = {"g0_train":'g0_train', "g1_train": 'g1_train', "g2_train": 'g2_train',
                   "g3_train": 'g3_train', "g0_test": 'g0_test',
                   "g1_test": 'g1_test',"g2_test": 'g2_test',
                   "g3_test": 'g3_test', 'train':'train', 'test':'test', 'val':'val'}

    dataset = WaterbirdsDataset(version='1.0', root_dir=data_dir, download=True,
                          split_scheme=split_scheme, split_names=split_names,
                                metadata_name='metadata_5g_v2.csv')
    # Get the training set

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
    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    use_cuda = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    num_groups = 4
    models = []
    data_train_set = [training0_data, training1_data, training2_data, training3_data]
    data_unseen_set = [test0_data, test1_data, test2_data, test3_data]
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size, shuffle=True, **kwargs)
    model = resnet50()
    train_gdro = False
    if train_gdro:
        training0_data.pseudo_group_label = 0
        training1_data.pseudo_group_label = 1
        training2_data.pseudo_group_label = 2
        training3_data.pseudo_group_label = 3
        dataset_train = torch.utils.data.ConcatDataset(data_train_set)
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=batch_size, shuffle=True,**kwargs)

        proportion_correct_train, proportion_correct_test = train_batched(
            model=model, weight_decay=weight_decay,
            dataloader=dataloader_train, dataloader_test=test_loader,
            wandb=wandb, label_wb = 'gdro given groups ',
            lr=lr, epochs=num_epochs, gdro=True,num_groups=4)

        print('Train data gdro given groups')
        print('proportion correct train given groups', proportion_correct_train)
        print('proportion correct test given groups', proportion_correct_test)
        test_acc_final = test_per_group(test_loader, model, dataset, 'gdro given groups ')
        for label, item in test_acc_final.items():
            wandb.summary[label] = item
    train_gdro_infered_g=False

    if train_gdro_infered_g:
        training0_data.pseudo_group_label = 0
        training1_data.pseudo_group_label = 1
        training2_data.pseudo_group_label = 1
        training3_data.pseudo_group_label = 1
        dataset_train = torch.utils.data.ConcatDataset(data_train_set)
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=batch_size, shuffle=True,**kwargs)

        proportion_correct_train, proportion_correct_test = train_batched(
            model=model, weight_decay=weight_decay,
            dataloader=dataloader_train, dataloader_test=test_loader,
            wandb=wandb, label_wb = 'gdro inferred groups',
            lr=lr, epochs=num_epochs, gdro=True, num_groups=2)

        print('Train data gdro given groups')
        print('proportion correct train given groups', proportion_correct_train)
        print('proportion correct test given groups', proportion_correct_test)
        test_acc_final = test_per_group(test_loader, model, dataset, 'gdro infered groups ')
        for label, item in test_acc_final.items():
            wandb.summary[label] = item

            
    train_and_cross_test = False
    min_data_len =  min([len(data) for data in data_train_set])
    result_matrix = np.zeros((num_groups, num_groups))
    if train_and_cross_test:
        for data_train, data_unseen in zip(data_train_set, data_unseen_set):
            if model == 'convnet':
                model = ConvNet()
            elif model == 'wide':
                model = wideresnet50()
            else:
                model = resnet50()
            size_data = len(data_train)
            data_indexes = random.sample([i for i in range(size_data)], min_data_len)
            data_train_sampled = torch.utils.data.Subset(data_train, data_indexes)
            dataloader_train = torch.utils.data.DataLoader(data_train_sampled, batch_size=batch_size,
                                                           shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(data_unseen, batch_size=batch_size,**kwargs)
            proportion_correct_train, proportion_correct_test = train_batched(
                model=model,weight_decay=weight_decay,
                dataloader=dataloader_train, dataloader_test=test_loader,
                lr=lr, epochs=num_epochs, wandb=wandb)
            print('proportion correct train', proportion_correct_train)
            print('proportion correct test', proportion_correct_test)
            test_per_group(test_loader, model, dataset, 'for matrix')
            models.append(model)

        for i, model in enumerate(models):
            for j, data_unseen in enumerate(data_unseen_set):
                dataloader_test = torch.utils.data.DataLoader(
                    data_unseen, batch_size=batch_size, **kwargs)
                correct = test_batched(model, dataloader_test, device)
                result_matrix[i,j] = correct
    
    # the following groups should be grouped together
    print(result_matrix)
    groups = provide_groupings_test_acc(result_matrix, k=0.10)
    # train model with same data but weight according to assigned groups
    groups = [{1,2,3},{0}]
    print(groups)
    data_train_set = [training0_data, training1_data, training2_data, training3_data]    
    weights = provide_group_weights(groups, data_train_set)
    print(weights)

    print('Models trained, testing models')
    if debug == True:
        num_epochs = num_epochs

    # test n models on n test groups

    # train model with equal weights across given groups.
    if model == 'convnet':
        model = ConvNet()
    elif model == 'wide':
        model = wideresnet50()
    else:
        model = resnet50()
    use_cuda = True
    print([len(data) for data in data_train_set])

    train_baseline = False
    train_method = True
    train_baseline_w = True
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size, shuffle=True, **kwargs)
    
    if train_baseline:
        dataset_train = torch.utils.data.ConcatDataset(data_train_set)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,**kwargs)

        proportion_correct_train, proportion_correct_test = train_batched(
            model=model, weight_decay=weight_decay,
            dataloader=dataloader_train, dataloader_test=test_loader,
            wandb=wandb, label_wb = 'no weights ', lr=lr, epochs=num_epochs)

        print('Train data without weights')
        print('proportion correct train given groups', proportion_correct_train)
        print('proportion correct test given groups', proportion_correct_test)
        test_acc_final = test_per_group(test_loader, model, dataset, 'no weights ')
        for label, item in test_acc_final.items():
            wandb.summary[label] = item
    # rows (i) contain different models
    # columns (j) contain different test groups
    # a[i,j] contains model i tested on test j
    train_method = True
    if train_method:
        if model == 'convnet':
            model = ConvNet()
        elif model == 'wide':
            model = wideresnet50()
        else:
            model = resnet50()
        weights_step1 = [[
            weight[1] for i in range(len(data))] for weight, data in zip(
                sorted(weights.items()),data_train_set)]
        weights_samples = [x for xs in weights_step1 for x in xs]
        dataset_train = torch.utils.data.ConcatDataset(data_train_set)
        data_size = calc_data_len_upsample(weights, data_train_set)
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights_samples,
                                                         num_samples=data_size,
                                                         replacement=True)

        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=batch_size, sampler=sampler,**kwargs)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size, shuffle=True, **kwargs)
        proportion_correct_train, proportion_correct_test = train_batched(
            model=model,weight_decay=weight_decay, wandb=wandb, label_wb='weighted by infered groups ',
            dataloader=dataloader_train, dataloader_test=test_loader, lr=lr, epochs=num_epochs)
        test_1 = test_batched(model, test_loader, device)
        print('weighted groups assigned by algorithm')
        print('proportion correct train assigned groups', proportion_correct_train)
        print('proportion correct test assigned groups', proportion_correct_test)
        test_acc_final = test_per_group(test_loader, model, dataset, 'weighted by infered groups ')
        print('proportion correct test1 assigned groups', test_1)
        for label, item in test_acc_final.items():
            wandb.summary[label] = item

    
    if train_baseline_w:
        if model == 'convnet':
            model = ConvNet()
        elif model == 'wide':
            model = wideresnet50()
        else:
            model = resnet50()

        weights_step1 = [[
            1/len(data) for i in range(len(data))] for data in data_train_set]
        weights_samples = [x for xs in weights_step1 for x in xs]
        weights = {key: w[0] for key, w in enumerate(weights_step1)}
        data_size = calc_data_len_upsample(weights, data_train_set)
        dataset_train = torch.utils.data.ConcatDataset(data_train_set)
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights_samples,
                                                         num_samples=data_size,
                                                         replacement=True)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                       sampler=sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size, **kwargs)

        proportion_correct_train, proportion_correct_test = train_batched(
            model=model,weight_decay=weight_decay, wandb=wandb, label_wb='weighted by given groups ',
            dataloader=dataloader_train, dataloader_test=test_loader, lr=lr, epochs=num_epochs)
        test_1 = test_batched(model, test_loader, device)
        print('Training data weighted according to given groups')
        print('proportion correct train given groups', proportion_correct_train)
        print('proportion correct test given groups', proportion_correct_test)
        test_acc_final = test_per_group(test_loader, model, dataset, 'weighted by given groups ')
        for label, item in test_acc_final.items():
            wandb.summary[label] = item

        print('proportion correct test1 given groups', test_1)
    


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data_dir', type=str,
                        default='/network/scratch/m/mizu.nishikawa-toomey/')
    parser.add_argument('--model', type=str,
                        default='resnet')
    
    args = parser.parse_args()

    main(**vars(args))
