from torchvision import transforms
import os
from datetime import datetime
import pickle
import numpy as np
import collections
import sys
import torch
import models
from active_learning_data import ActiveLearningDataSources
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
from acquisitions import (Random, UniformSources,
                          EntropyPerSource, AccuracyPerSource, Entropy,
                          EntropyUniformSources, MI,
                          EntropyPerSourceNLargest, EntropyPerSourceOrdered,
                          EntropyBatch)

from data_loading import (waterbirds_n_sources, celeba_n_sources,
                          cmnist_n_sources, camelyon17, camelyon17_ood, cmnist_n_sources_ood,
                          fmow, fmow_ood, cmnist_10_n_sources, camelyon17_2sources)
from torch.utils.data import ConcatDataset, DataLoader
import arguments

# to turn off wandb, export WANDB_MODE=disabled

def main(args):
    if args.acquisition == 'random_gdro':
        args.gdro = True
    config = getattr(arguments, args.mode)()
    args = arguments.populate_args_from_dataclass(args, config)
    baseline_methods = ['entropy', 'random', 'random_gdro']
    if args.acquisition in baseline_methods:
        args.start_acquisition = 'random'
    # argparse has priority over dataclass
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
    true_group_in_loss = False
    group_string_map_test = None
    if args.data_mode == 'wb':
        dataset, training_data_dict, val_data_dict, test_data_dict = waterbirds_n_sources(args.num_minority_points,
                                                                           args.num_majority_points,
                                                                           n_maj_sources=args.n_maj_sources,
                                                                           metadata_path='metadata_larger.csv',
                                                                           root_dir="/network/scratch/m/"
                                                                           "mizu.nishikawa-toomey/waterbird_larger")
        num_groups = 4
        num_sources = args.n_maj_sources + 1
    if args.data_mode == 'celeba':
        dataset, training_data_dict, val_data_dict, test_data_dict = celeba_n_sources(args.num_minority_points,
                                                                       args.num_majority_points, args.n_maj_sources)
        num_groups = 4
        num_sources = args.n_maj_sources 
    if args.data_mode == 'cmnist':
        dataset, training_data_dict, val_data_dict, test_data_dict = cmnist_n_sources(args.num_minority_points, args.num_majority_points,
                                                                       n_maj_sources=args.n_maj_sources,
                                                                       causal_noise=args.causal_noise,
                                                                       spurious_noise=args.spurious_noise,
                                                                       num_digits_per_target=args.num_digits_per_target,
                                                                       binary_classification=args.binary_classification)
        num_sources = args.n_maj_sources + 1
        num_groups = 4
    if args.data_mode == 'cmnist_10':
        dataset, training_data_dict, val_data_dict, test_data_dict = cmnist_10_n_sources(args.num_minority_points, args.num_majority_points,
                                                                                         n_maj_sources=args.n_maj_sources,
                                                                                         causal_noise=args.causal_noise,
                                                                                         spurious_noise=args.spurious_noise,
                                                                                         num_digits_per_target=1,
                                                                                         binary_classification=False)
        num_sources = args.n_maj_sources + 1
        num_groups = 20

    if args.data_mode == 'cmnist_ood':
        dataset, training_data_dict, val_data_dict, test_data_dict = cmnist_n_sources_ood(args.num_minority_points, args.num_majority_points,
                                                                           n_maj_sources=args.n_maj_sources,
                                                                           causal_noise=args.causal_noise,
                                                                           spurious_noise=args.spurious_noise,
                                                                           num_digits_per_target=args.num_digits_per_target,
                                                                           binary_classification=args.binary_classification)
        num_groups = args.n_maj_sources
        dataset.set_group_string_map_test({'y0r_y1g': 'y01_y1g'})

    if args.data_mode == 'camelyon':
        source_proportions = args.source_proportions
        if source_proportions is None:
            source_proportions = np.random.dirichlet(np.ones(5))
        print(source_proportions)
        dataset, training_data_dict, val_data_dict, test_data_dict = camelyon17(max_training_data_size=6000, source_proportions=source_proportions)
        source_string_map = {str(key): key for key, value in training_data_dict.items()}
        dataset.set_source_string_map(source_string_map)
        num_groups = 5
        num_sources = 5

    if args.data_mode == 'camelyon2s':
        source_proportions = args.source_proportions
        if source_proportions is None:
            source_proportions = np.random.dirichlet(np.ones(5))
        print(source_proportions)
        dataset, training_data_dict, val_data_dict, test_data_dict = camelyon17_2sources(max_training_data_size=6000, source_proportions=source_proportions)
        source_string_map = {str(key): key for key, value in training_data_dict.items()}
        dataset.set_source_string_map(source_string_map)
        num_groups = 5
        num_sources = 2

    if args.data_mode == 'camelyon_ood':
        source_proportions = args.source_proportions
        if len(source_proportions) == 1:
            source_proportions = np.random.dirichlet(np.ones(4))
        print(source_proportions)
        dataset, training_data_dict, test_data_dict = camelyon17_ood(max_training_data_size=6000,
                                                                     source_proportions=source_proportions,
                                                                     test_source=args.test_source)
        
        source_string_map = {str(key): key for key, value in training_data_dict.items()}
        dataset.set_source_string_map(source_string_map)
        dataset.set_source_string_map_test({str(args.test_source): args.test_source})
        num_sources = 4
        num_sources = 4
        group_string_map_test = dataset.group_string_map_test

    if args.data_mode == 'fmow':
        source_proportions = args.source_proportions
        print(source_proportions)
        to_log.update({'source_proportions': source_proportions})
        dataset, training_data_dict, test_data_dict = fmow(max_training_data_size=args.max_training_data_size,
                                                           source_proportions=source_proportions)
        #source_string_map = {str(key): key for key, value in training_data_dict.items()}
        #dataset.set_source_string_map(source_string_map)

    if args.data_mode == 'fmow_ood':
        source_proportions = args.source_proportions
        if len(source_proportions) == 1:
            source_proportions = np.random.dirichlet(np.ones(4))
        print(source_proportions)
        to_log.update({'source_proportions': source_proportions})
        dataset, training_data_dict, test_data_dict = fmow_ood(max_training_data_size=args.max_training_data_size,
                                                               source_proportions=source_proportions,
                                                               test_source=args.test_source)
        source_string_map = {str(key): key for key, value in training_data_dict.items()}
        dataset.set_source_string_map(source_string_map)
        dataset.set_source_string_map_test({str(args.test_source): args.test_source})
        num_sources = 4
        source_string_map_test = dataset.source_string_map_test

        #source_string_map = {str(key): key for key, value in training_data_dict.items()}
        #dataset.set_source_string_map(source_string_map)


    print("data loaded")
    num_sources = len(training_data_dict)
    samples_per_source = int(args.al_size / num_sources)
    
    source_dict_uniform_sources = {key: samples_per_source for key in range(num_sources)}
    #source_dict_uniform_sources = {0: int(args.min_prop*args.al_size), 1: int((1-args.min_prop)*args.al_size)}
    al_data = ActiveLearningDataSources([*training_data_dict.values()],
                                       [*val_data_dict.values()],
                                       [*test_data_dict.values()], 2, args.batch_size,
                                       batch_size_test=args.batch_size_test)
    
    method_map = {
        'random': Random(al_data, args.al_size),
        'random_gdro': Random(al_data, args.al_size),
        'uniform_sources': UniformSources(al_data, source_dict_uniform_sources, args.al_size),
        'entropy_per_source': EntropyPerSource(al_data=al_data, al_size=args.al_size,
                                             temperature=args.temperature, num_sources=num_sources),
        'entropy_per_source_soft_rank': EntropyPerSource(al_data=al_data, al_size=args.al_size,
                                             temperature=args.temperature, num_sources=num_sources,
                                             within_source_acquisition='softrank'),
        'entropy_per_source_soft_max': EntropyPerSource(al_data=al_data, al_size=args.al_size,
                                             temperature=args.temperature, num_sources=num_sources,
                                             within_source_acquisition='softmax'),
        'entropy_per_source_power': EntropyPerSource(al_data=al_data, al_size=args.al_size,
                                             temperature=args.temperature, num_sources=num_sources,
                                             within_source_acquisition='power'),
        'entropy_per_source_top_k': EntropyPerSource(al_data=al_data, al_size=args.al_size,
                                                   temperature=args.temperature, num_sources=num_sources,
                                                   within_source_acquisition='topk'),
        'n_largest_soft_rank': EntropyPerSourceNLargest(al_data=al_data, al_size=args.al_size,
                                                       num_sources=num_sources, n=args.m_sources_size,
                                             within_source_acquisition='softrank'),
        'n_largest_soft_max': EntropyPerSourceNLargest(al_data=al_data, al_size=args.al_size,
                                                      num_sources=num_sources, n=args.m_sources_size,
                                             within_source_acquisition='softmax'),
        'n_largest_power': EntropyPerSourceNLargest(al_data=al_data, al_size=args.al_size,
                                                   num_sources=num_sources, n=args.m_sources_size,
                                             within_source_acquisition='power'),
        'n_largest_top_k': EntropyPerSourceNLargest(al_data=al_data, al_size=args.al_size,
                                                   num_sources=num_sources, n=args.m_sources_size,
                                                   within_source_acquisition='topk'),

        'mi_per_source': EntropyPerSource(al_data=al_data, al_size=args.al_size,
                                             temperature=args.temperature, num_sources=num_sources, mi=True),
        'entropy': Entropy(al_data=al_data, al_size=args.al_size),
        'mi': MI(al_data=al_data, al_size=args.al_size),
        'entropy_uniform_sources': EntropyUniformSources(al_data=al_data, al_size=args.al_size),
        'entropy_per_source_n_largest': EntropyPerSourceNLargest(al_data=al_data,
                                                                 al_size=args.al_size,
                                                                 n=args.m_sources_size,
                                                                 num_sources=num_sources),
        'entropy_per_point_soft_rank': EntropyBatch(al_data=al_data,
                                                    al_size=args.al_size, mode='softrank'),
        'entropy_per_point_soft_max': EntropyBatch(al_data=al_data,
                                                   al_size=args.al_size, mode='softmax'),
        'entropy_per_point_power': EntropyBatch(al_data=al_data, al_size=args.al_size, mode='power')}
        

    


    mi = False
    if args.acquisition == 'mi':
        mi = True
    if args.acquisition == 'random_gdro':
        args.gdro = True
    # initial random or uniform acquisition to start with
    acquisition_method = method_map[args.start_acquisition]
    indices = acquisition_method.return_indices()
    al_data.acquire_with_indices(indices)
    now = datetime.now()
    time = now.strftime("%A, %B %d, %Y %H:%M:%S")
    path = f"/network/scratch/m/mizu.nishikawa-toomey/checkpoints/{args.mode}_{args.acquisition}_{args.seed}"
    model_checkpoint_path = path + time + 'model.pt'

    for i in range(1, args.al_iters):
        print('al iteration: ', i)
        # setting up trainig
        
        acquisition_method = method_map[args.acquisition]
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = getattr(models, args.model_name)
        model = model(args.num_classes, args.pretrained, args.frozen_weights)
        dataloader_train, dataloader_val, dataloader_test = al_data.get_train_and_test_and_val_loader(
            batch_size=args.batch_size)
        num_points = len(al_data.train.indices)

        proportion_correct_train, proportion_correct_val, proportion_correct_test, groups_in_train, sources_in_train, wga, wga_test = train_batched(
            model=model, dataloader=dataloader_train, dataloader_val=dataloader_val,
            dataloader_test=dataloader_test, lr=args.lr, num_epochs=args.num_epochs,
            num_groups=num_groups, num_sources=num_sources, weight_decay=args.weight_decay,
            group_mapping_fn=dataset.group_mapping_fn, gdro=args.gdro,
            group_string_map=dataset.group_string_map, 
            group_string_map_test=group_string_map_test,
            model_checkpoint_path=model_checkpoint_path,
            true_group_in_loss=true_group_in_loss, sample_batch_val=args.num_batch_val_samples)
        
        # log training on wandb
        to_log.update({'train_acc': proportion_correct_train,
                       'num points':num_points,
                       'wga': wga, 'wga test': wga_test})

        # IDT FIX acquisition imports
        # calculate score over pool depending on acquisition method
        to_log_acq = acquisition_method.information_for_acquisition(model)
        if to_log_acq != None:    
            to_log.update({'average entropy for source ' + str(key) : value for key, value in to_log_acq.items()})
        
        # acquire data
        indices = acquisition_method.return_indices()
        al_data.acquire_with_indices(indices)

        # compute metrics and logging for debugging
        for source_name, data in test_data_dict.items():
            score_test = calc_ent_per_point_batched(
                model, DataLoader(data, batch_size=args.batch_size, drop_last=True), mean=True, mi=mi,
                sampled_batches=args.num_batch_val_samples)
            to_log.update({f'ent {source_name}': score_test})
        to_log.update({f"sources {key}  in train" : value for key, value in sources_in_train.items()})
        to_log.update({f"source {key}  in train" : value for key, value in sources_in_train.items()})
        to_log.update(proportion_correct_test)
        to_log.update(proportion_correct_val)
        wandb.log(to_log) 
        pprint(to_log)
        long_term_log = log_dict(log_term_log, to_log)
    plot_dictionary(to_log)
    os.remove(model_checkpoint_path)
    return to_log

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='wb')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--m_sources_size', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--causal_noise', type=float, default=0)
    parser.add_argument('--spurious_noise', type=float, default=0)
    parser.add_argument('--acquisition', type=str, default='random')
    parser.add_argument('--start_acquisition', type=str, default='uniform_sources')
    parser.add_argument('--project_name', type=str, default='test')
    parser.add_argument('--gdro', default=False, action='store_true')
    parser.add_argument('--frozen_weights', default=False, action='store_true')
    parser.add_argument('--pretrained', default=False, action='store_true')

    parser.add_argument('--num_digits_per_target', type=int, default=None)
    parser.add_argument('--max_training_data_size', type=int, default=None)
    parser.add_argument('--min_prop', type=float, default=None)
    parser.add_argument('--num_batch_val_samples', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--source_proportions', type=float, nargs='+', default=None)
    parser.add_argument('--data_mode', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--batch_size_test', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--num_majority_points', type=int, default=None)
    parser.add_argument('--num_minority_points', type=int, default=None)
    parser.add_argument('--al_iters', type=int, default=None)
    parser.add_argument('--al_size', type=int, default=None)
    parser.add_argument('--n_maj_sources', type=int, default=None)
    parser.add_argument('--test_source', type=int, default=None)
    
    args = parser.parse_args()

    main(args)
