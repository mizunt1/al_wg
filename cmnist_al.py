from torchvision import transforms
import random
import numpy as np
import collections
import torch
from models import BayesianNet
from active_learning_data import ActiveLearningDataGroups
from cmnist_ram import ColoredMNISTRAM, train_batched, test_batched
from tools import calc_ent_batched, calc_ent_per_group_batched, plot_dictionary, log_dict
from pprint import pprint
from acquisitions import (Random, UniformGroups,
                          EntropyPerGroup, AccuracyPerGroup, Entropy, EntropyUniformGroups)
import wandb
from tools import slurm_infos
from create_datasets import two_groups_cmnist, five_groups_cmnist, ten_groups_cmnist,ten_groups_cmnist_multiple_int, yxm_groups_cmnist, groups_to_env, leaky_groups, one_balanced_cmnist

# to turn off wandb, export WANDB_MODE=disabled
def main(seed, project_name='al_wg_test', al_iters=10, al_size=100, num_epochs=150,
         acquisition='random', data1_size=5000,
         data2_size=1000, start_acquisition='random',
         data_mode='two_groups', causal_noise=0, spurious_noise=0, num_spurious_groups=4):
    wandb.init(
        project=project_name,
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    wandb.run.summary.update(slurm_infos())

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
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
    if data_mode == 'yxm_groups':
        data_train, start_idx = yxm_groups_cmnist(trans)
        group_to_log1 = 0
        group_to_log2 = 1
    elif data_mode == 'groups_to_env':
        data_train, start_idx = groups_to_env(trans)
        group_to_log1 = 0
        group_to_log2 = 1
    elif data_mode == 'five_groups':        
        data_train, start_idx = five_groups_cmnist(spurious_noise, causal_noise, data1_size, data2_size, trans)
        group_to_log1 = 0
        group_to_log2 = 4
    elif data_mode=='two_groups':
        data_train, start_idx = two_groups_cmnist(spurious_noise, causal_noise,data1_size, data2_size, trans)
        group_to_log1 = 0
        group_to_log2 = 1
    elif data_mode == 'ten_groups':
        data_train, start_idx = ten_groups_cmnist(spurious_noise, causal_noise,data1_size, data2_size, trans)
        group_to_log1 = 0
        group_to_log2 = 9
    elif data_mode == 'ten_groups_multiple_int':
        data_train, start_idx = ten_groups_cmnist_multiple_int(spurious_noise,
                                                               causal_noise,
                                                               data1_size,
                                                               data2_size,
                                                               trans)    
    elif data_mode == 'leaky_groups':
        data_train, start_idx = leaky_groups(data1_size, sp_noise=spurious_noise,
                                             num_spurious_groups=num_spurious_groups, trans=trans)
        group_to_log1 = 0
        group_to_log2 = num_spurious_groups

    elif data_mode == 'one_balanced_cmnist':
        data_train, start_idx = one_balanced_cmnist(data1_size,
                                                    num_spurious_groups=num_spurious_groups, trans=trans)
        group_to_log1 = 0
        group_to_log2 = num_spurious_groups

    else:
        print('data mode not recognised')

    num_groups = len(data_train)
    samples_per_group = int(al_size / num_groups)
    
    group_dict = {key: samples_per_group for key in range(num_groups)}
    dataset1_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                     causal_noise=0,
                                     transform=trans, start_idx=start_idx, num_samples=5000,
                                    flip_sp=False, group_idx=0)
    start_idx += 5000
    dataset2_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                     causal_noise=0,
                                     transform=trans, start_idx=start_idx, num_samples=5000,
                                      flip_sp=True, group_idx=1)
    start_idx += 5000
    dataloader1_unseen = torch.utils.data.DataLoader(dataset1_unseen,
                                                      batch_size=64, **kwargs)
    dataloader2_unseen = torch.utils.data.DataLoader(dataset2_unseen,
                                                      batch_size=64, **kwargs)
    dataset_test = ColoredMNISTRAM(root='./data',
                                   train=False, spurious_noise=0.5, 
                                   causal_noise=0.0,
                                   transform=trans,
                                   start_idx=0, 
                                   num_samples=5000,
                                   flip_sp=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True, **kwargs)
    data_causal = ColoredMNISTRAM(root='./data', train=False, spurious_noise=0.5, 
                                     causal_noise=0,
                                     transform=trans, start_idx=5000, num_samples=2000, 
                                     flip_sp=False)
    data_sp = ColoredMNISTRAM(root='./data', train=False, spurious_noise=0.0, 
                              causal_noise=0.5,
                              transform=trans, start_idx=7000, 
                              num_samples=2000, flip_sp=False)

    dataloader_causal = torch.utils.data.DataLoader(data_causal,
                                                    batch_size=64, **kwargs)
    dataloader_sp = torch.utils.data.DataLoader(data_sp,
                                                batch_size=64, **kwargs)

    al_data = ActiveLearningDataGroups(data_train, dataset_test, 2)
    method_map = {
        'random': Random,
        'uniform_groups': UniformGroups,
        'entropy_per_group': EntropyPerGroup,
        'accuracy': AccuracyPerGroup,
        'entropy': Entropy,
        'entropy_uniform_groups': EntropyUniformGroups}
    kwargs_map = {'random': {'al_data': al_data, 'al_size': al_size},
                  'uniform_groups': {'al_data': al_data, 'group_proportions': group_dict},
                  'entropy_per_group': {'al_data': al_data, 'al_size': al_size},
                  'entropy': {'al_data': al_data, 'al_size': al_size},
                  'accuracy': {'al_data': al_data, 'al_size': al_size},
                  'entropy_uniform_groups':{'al_data': al_data, 'al_size': al_size,
                                            'group_proportions': group_dict}}
    
    # initial random or uniform acquisition to start with
    acquisition_method = method_map[start_acquisition](**kwargs_map[start_acquisition])
    indices = acquisition_method.return_indices()
    al_data.acquire_with_indices(indices)

    if acquisition == 'accuracy':
        # first we have training on random acquisiion
        # when using cross entropy based acquisition, we require 
        # two uniform group acquisitions
        model = BayesianNet(num_classes=2)
        dataloader_train, dataloader_test = al_data.get_train_and_test_loader(batch_size=64)
        # train model1 on first batch D1
        proportion_correct_train, proportion_correct_test, group_dict_train = train_batched(
            model=model, dataloader=dataloader_train,
            dataloader_test=dataloader_test, lr=0.001,
            epochs=num_epochs, num_groups=num_groups)
        ent1, cross_ent1 = calc_ent_batched(model, dataloader1_unseen, num_models=100)
        ent2, cross_ent2 = calc_ent_batched(model, dataloader2_unseen, num_models=100)
        causal_correct = test_batched(model, dataloader_causal)
        sp_correct = test_batched(model, dataloader_sp)
        num_points = len(al_data.train.indices)
        to_log = {'train_acc': proportion_correct_train, 'test_acc': proportion_correct_test,
                  'cross_ent_1': cross_ent1, 'cross_ent_2': cross_ent2,
                  'num points':num_points, 'ent1': ent1, 'ent2' :ent2,
                  'causal acc':causal_correct, 'sp acc': sp_correct,
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
        model = BayesianNet(num_classes=2)
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
        elif acquisition == 'entropy_uniform_groups':
            acquisition_method.information_for_acquisition(model, num_groups)
        else:
            print('acquisition not recognised')
        # acquire data
        indices = acquisition_method.return_indices()
        al_data.acquire_with_indices(indices)
        # compute metrics and logging
        ent1, cross_ent1 = calc_ent_batched(model, dataloader1_unseen, num_models=100)
        ent2, cross_ent2 = calc_ent_batched(model, dataloader2_unseen, num_models=100)
        causal_correct = test_batched(model, dataloader_causal)
        sp_correct = test_batched(model, dataloader_sp)
        to_log = {'train_acc': proportion_correct_train, 'test_acc': proportion_correct_test,
                  'cross_ent_1': cross_ent1, 'cross_ent_2': cross_ent2,
                  'num points':num_points, 'ent1': ent1, 'ent2' :ent2,
                  'causal acc':causal_correct, 'sp acc': sp_correct,
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
    parser.add_argument('--num_spurious_groups', type=int, default=4)
    parser.add_argument('--acquisition', type=str, default='random')
    parser.add_argument('--start_acquisition', type=str, default='random')
    parser.add_argument('--project_name', type=str, default='al_wg')
    parser.add_argument('--data_mode', type=str, default='two_groups')
    parser.add_argument('--causal_noise', type=float, default=0)
    parser.add_argument('--spurious_noise', type=float, default=0)
    
    args = parser.parse_args()

    main(**vars(args))
    seed = 0
