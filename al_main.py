from torchvision import transforms
import pickle
import numpy as np
import collections
import sys
import torch
import models
from active_learning_data import ActiveLearningDataGroups
from tools import calc_ent_batched, calc_ent_per_point_batched, plot_dictionary, log_dict
from pprint import pprint
import wandb
from torch.utils.data import DataLoader, ConcatDataset, Subset
import random
from tools import slurm_infos
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from waterbirds_dataset import WaterbirdsDataset
from trainer import train_batched, test_batched
from acquisitions import (Random, UniformGroups,
                          EntropyPerGroup, AccuracyPerGroup, Entropy,
                          EntropyUniformGroups, MI, EntropyPerGroupNLargest, EntropyPerGroupOrdered)
from data_loading import (waterbirds, waterbirds_n_sources, celeba, celeba_n_sources,
                          cmnist_n_sources, iwildcam_n_sources, camelyon17)
from torch.utils.data import ConcatDataset, DataLoader

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
    to_log = collections.defaultdict(list)
    log_term_log = collections.defaultdict(list)
    print("loading data")
    model = getattr(models, args.model_name)

    if args.data_mode == 'wb':
        if args.data_wo_sources:
            dataset, training_data_dict, test_data_dict = waterbirds(args.num_minority_points,
                                                            args.num_majority_points,
                                                            metadata_path='metadata_larger.csv',
                                                            root_dir="/network/scratch/m/"
                                                            "mizu.nishikawa-toomey/waterbird_larger")
                                                            
            true_group_in_loss = True
        else:
            dataset, training_data_dict, test_data_dict = waterbirds_n_sources(args.num_minority_points,
                                                                      args.num_majority_points,
                                                                      n_maj_sources=args.n_maj_sources,
                                                                      metadata_path='metadata_larger.csv',
                                                                      root_dir="/network/scratch/m/"
                                                                      "mizu.nishikawa-toomey/waterbird_larger")
            true_group_in_loss = False
        model = model(2, args.pretrained, args.frozen_weights)
    if args.data_mode == 'celeba':
        if args.data_wo_sources:
            dataset, training_data_dict, test_data_dict = celeba(args.num_minority_points,
                                                                 args.num_majority_points)
            true_group_in_loss = True
        else:
            dataset, training_data_dict, test_data_dict = celeba_n_sources(args.num_minority_points,
                                                                           args.num_majority_points, args.n_maj_sources)
            true_group_in_loss = False
    if args.data_mode == 'cmnist':
        dataset, training_data_dict, test_data_dict = cmnist_n_sources(args.num_minority_points, args.num_majority_points,
                                                                       n_maj_sources=args.n_maj_sources,
                                                                       causal_noise=args.causal_noise,
                                                                       spurious_noise=args.spurious_noise,
                                                                       num_digits_per_target=args.num_digits_per_target)
        true_group_in_loss = False
        model = model(2, args.pretrained, args.frozen_weights)
    if args.data_mode == 'iwildcam':
        dataset, training_data_dict, test_data_dict = iwildcam_n_sources(n_sources=args.n_maj_sources)
        true_group_in_loss = True
        model = model(2, args.pretrained, args.frozen_weights)

    if args.data_mode == 'camelyon':
        group_proportions = [0.70, 0.1, 0.1, 0.1]
        print(group_proportions)
        dataset, training_data_dict, test_data_dict = camelyon17(max_training_data_size=6000, group_proportions=group_proportions)
        true_group_in_loss = False
        group_string_map = {str(key): key for key, value in training_data_dict.items()}
        dataset.set_group_string_map(group_string_map)
    num_groups = args.n_maj_sources + 1
    print("data loaded")
    num_sources = len(training_data_dict)
    samples_per_group = int(args.al_size / num_groups)
    group_dict_uniform_groups = {key: samples_per_group for key in range(num_groups)}
    al_data = ActiveLearningDataGroups([*training_data_dict.values()], [*test_data_dict.values()], 2, args.batch_size)

    method_map = {
        'random': Random(al_data, args.al_size),
        'uniform_groups': UniformGroups(al_data, group_dict_uniform_groups, args.al_size),
        'entropy_per_group': EntropyPerGroup(al_data=al_data, al_size=args.al_size,
                                             temperature=args.temperature, num_groups=num_groups),
        'entropy': Entropy(al_data=al_data, al_size=args.al_size),
        'mi': MI(al_data=al_data, al_size=args.al_size),
        'entropy_uniform_groups': EntropyUniformGroups(al_data=al_data, al_size=args.al_size),
        'entropy_per_group_n_largest': EntropyPerGroupNLargest(al_data=al_data, al_size=args.al_size, n=args.n_groups_size, num_groups=num_groups)}
    mi = False
    if args.acquisition == 'mi':
        mi = True

    # initial random or uniform acquisition to start with
    acquisition_method = method_map[args.start_acquisition]
    indices = acquisition_method.return_indices()
    al_data.acquire_with_indices(indices)
        
    for i in range(1, args.al_iters):
        print('al iteration: ', i)
        # setting up trainig
        
        acquisition_method = method_map[args.acquisition]
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = getattr(models, args.model_name)
        model = model(2, args.pretrained, args.frozen_weights)
        dataloader_train, dataloader_test = al_data.get_train_and_test_loader(
            batch_size=args.batch_size)
        num_points = len(al_data.train.indices)
        proportion_correct_train, proportion_correct_test, groups_in_train, sources_in_train, wga = train_batched(
            model=model, dataloader=dataloader_train,
            dataloader_test=dataloader_test, lr=args.lr, num_epochs=args.num_epochs,
            num_groups=num_groups, num_sources=num_sources, weight_decay=args.weight_decay,
            group_mapping_fn=dataset.group_mapping_fn, gdro=args.gdro,
            group_string_map=dataset.group_string_map,
            true_group_in_loss=true_group_in_loss, sample_batch_test=args.num_batch_test_samples)
        
        # log training
        to_log.update({'train_acc': proportion_correct_train,
                       'num points':num_points,
                       'wga': wga})

        # IDT FIX acquisition imports
        # calculate score over pool depending on acquisition method
        to_log_acq = acquisition_method.information_for_acquisition(model)
        if to_log_acq != None:    
            to_log.update({'average entropy for source ' + str(key) : value for key, value in to_log_acq.items()})
        
        # acquire data
        indices = acquisition_method.return_indices()
        al_data.acquire_with_indices(indices)

        # compute metrics and logging for debugging
        for group_name, data in test_data_dict.items():
            score_test = calc_ent_per_point_batched(
                model, DataLoader(data, batch_size=args.batch_size, drop_last=True), mean=True, mi=mi,
                sampled_batches=args.num_batch_test_samples)
            to_log.update({f'ent {group_name}': score_test})
        to_log.update({f"groups {key}  in train" : value for key, value in groups_in_train.items()})
        to_log.update({f"source {key}  in train" : value for key, value in sources_in_train.items()})
        to_log.update(proportion_correct_test)
        wandb.log(to_log) 
        pprint(to_log)
        long_term_log = log_dict(log_term_log, to_log)
    plot_dictionary(to_log)
    return to_log

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_minority_points', type=int, default=400)
    parser.add_argument('--num_batch_test_samples', type=int, default=100)
    parser.add_argument('--num_majority_points', type=int, default=4000)
    parser.add_argument('--al_iters', type=int, default=20)
    parser.add_argument('--al_size', type=int, default=30)
    parser.add_argument('--n_maj_sources', type=int, default=3)
    parser.add_argument('--n_groups_size', type=int, default=1)
    parser.add_argument('--num_digits_per_target', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--size', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--causal_noise', type=float, default=0)
    parser.add_argument('--spurious_noise', type=float, default=0)
    parser.add_argument('--acquisition', type=str, default='random')
    parser.add_argument('--data_mode', type=str, default='wb')
    parser.add_argument('--model_name', type=str, default='BayesianNetRes50ULarger')
    parser.add_argument('--start_acquisition', type=str, default='uniform_groups')
    parser.add_argument('--project_name', type=str, default='test')
    parser.add_argument('--gdro', default=False, action='store_true')
    parser.add_argument('--train_all_data', default=False, action='store_true')
    parser.add_argument('--balanced', default=False, action='store_true')
    parser.add_argument('--maj_group_only', default=False, action='store_true')
    parser.add_argument('--frozen_weights', default=False, action='store_true')
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--data_wo_sources', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
