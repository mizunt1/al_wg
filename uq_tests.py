import wandb
from tools import calc_ent_per_point_batched
from tools import slurm_infos
import numpy as np
import torch
import collections
import models
import pandas as pd
import os
from celeba import CelebA
from torchvision import transforms
from waterbirds_dataset import WaterbirdsDataset
from celeba import CelebA
from trainer import train_batched
from torch.utils.data import DataLoader
from mc_dropout import set_dropout_p
from torch.utils.data import ConcatDataset, DataLoader
from data_loading import waterbirds, celeba, cmnist_n_sources, iwildcam_n_sources, camelyon17

def main(args):
    wandb.init(
        project=args.project_name,
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    wandb.run.summary.update(slurm_infos())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    results = pd.DataFrame()
    log = collections.defaultdict(list)
    #datasize = [1000, 2000]
    #datasize = [50, 100, 150, 450, 500, 1200, 1400, 2000]
    datasize = [50, 100, 200, 500]
    for size in datasize:
        num_minority_points = int(size*args.minority_prop)
        num_majority_points = size-num_minority_points
        print('size '+ str(size))
        model = getattr(models, args.model_name)

        if 'clip' in args.model_name.lower():
            model = model.float()
            img_size = 244
        else:
            img_size = None
        if args.data_mode == 'wb':
            dataset, training_data_dict, test_data_dict = waterbirds(num_minority_points,
                                                                     num_majority_points,
                                                                     metadata_path='metadata_larger.csv',
                                                                     root_dir="/network/scratch/m/"
                                                                     "mizu.nishikawa-toomey/waterbird_larger", img_size=img_size)
            ood_dataset, ood_training_data_dict, ood_test_data_dict = celeba(num_minority_points,
                                                                             num_majority_points,
                                                                             batch_size=args.batch_size,img_size=img_size)
            true_group_in_loss = True
        if args.data_mode == 'celeba':
            dataset, training_data_dict, test_data_dict = celeba(num_minority_points,
                                                                 num_majority_points,
                                                                 batch_size=args.batch_size,
                                                                 img_size=img_size)
                                                                 
            ood_dataset, ood_training_data_dict, ood_test_data_dict = waterbirds(num_minority_points,
                                                                                 num_majority_points,
                                                                                 metadata_path='metadata_larger.csv',
                                                                                 img_size=img_size,
                                                                                 root_dir="/network/scratch/m/"
                                                                                 "mizu.nishikawa-toomey/waterbird_larger")
            true_group_in_loss = True
        # train model
        if args.data_mode == 'cmnist':
            dataset, training_data_dict, test_data_dict = cmnist_n_sources(num_minority_points, num_majority_points,
                                                                           n_maj_sources=args.n_maj_sources)
            ood_dataset, ood_training_data_dict, ood_test_data_dict = waterbirds(num_minority_points,
                                                                                 num_majority_points,
                                                                                 metadata_path='metadata_larger.csv',
                                                                                 img_size=128,
                                                                                 root_dir="/network/scratch/m/"
                                                                                 "mizu.nishikawa-toomey/waterbird_larger")
            true_group_in_loss = True

        if args.data_mode == 'iwildcam':
            dataset, training_data_dict, test_data_dict = iwildcam_n_sources(n_sources=args.n_maj_sources, max_training_data_size=size)
            ood_dataset, ood_training_data_dict, ood_test_data_dict = waterbirds(num_minority_points,
                                                                                 num_majority_points,
                                                                                 metadata_path='metadata_larger.csv',
                                                                                 img_size=128,
                                                                                 root_dir="/network/scratch/m/"
                                                                                 "mizu.nishikawa-toomey/waterbird_larger")
            group_string_map = {key: key for key, value in training_data_dict.items()}
            dataset.set_group_string_map(group_string_map)
            true_group_in_loss = True

        if args.data_mode == 'camelyon':
            group_proportions = [0.25, 0.33, 0.02, 0.4]
            dataset, training_data_dict, test_data_dict = camelyon17(max_training_data_size=size, group_proportions=group_proportions)
            ood_dataset, ood_training_data_dict, ood_test_data_dict = waterbirds(num_minority_points,
                                                                                 num_majority_points,
                                                                                 metadata_path='metadata_larger.csv',
                                                                                 img_size=128,
                                                                                 root_dir="/network/scratch/m/"
                                                                                 "mizu.nishikawa-toomey/waterbird_larger")
            group_string_map = {str(key): key for key, value in training_data_dict.items()}
            dataset.set_group_string_map(group_string_map)
            true_group_in_loss = True

        model = model(2, args.pretrained, args.frozen_weights)
        num_groups = len(training_data_dict)
        training_loader = DataLoader(ConcatDataset([*training_data_dict.values()]), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(ConcatDataset([*test_data_dict.values()]), batch_size=args.batch_size, shuffle=True)
        ood_test_loader = DataLoader(ConcatDataset([*ood_test_data_dict.values()]), batch_size=args.batch_size, shuffle=True)
        proportion_correct_train, proportion_correct_test, groups_in_train, wga = train_batched(
            model=model, dataloader=training_loader,
            dataloader_test=test_loader, lr=args.lr, num_epochs=args.num_epochs,
            num_groups=num_groups, weight_decay=args.weight_decay,
            group_mapping_fn=dataset.group_mapping_fn, gdro=args.gdro,
            group_string_map=dataset.group_string_map, true_group_in_loss=true_group_in_loss)
        log.update({'train_acc': proportion_correct_train,
                    'num points': size,
                    'wga': wga})
        log.update(proportion_correct_test)
        # calculate UQ metric 
        for group_name, data in test_data_dict.items():
            score_test = calc_ent_per_point_batched(
                model, DataLoader(data, batch_size=args.batch_size), mean=True)
            log.update({f'ent {group_name}': score_test})

        ood_entropy = calc_ent_per_point_batched(model, ood_test_loader, mean=True)
        log.update({'ood entropy': ood_entropy})
        wandb.log(log)
        results = pd.concat([pd.DataFrame(log, index=[0]), results],ignore_index=True)
    dir_name = f"{args.data_mode}/{args.save_dir}/{args.seed}"
    os.makedirs(dir_name, exist_ok=True) 
    results.to_csv(f"{dir_name}/{args.minority_prop}.csv")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--n_maj_sources', type=int, default=4)
    parser.add_argument('--frozen_weights', default=False, action='store_true')
    parser.add_argument('--pretrained', default=True, action='store_false')
    parser.add_argument('--gdro', default=False, action='store_true')
    parser.add_argument('--minority_prop', type=float, default=0.2)
    parser.add_argument('--mc_drop_p', type=float, default=None)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--model_name', type=str, default='BayesianNetRes50ULarger')
    parser.add_argument('--project_name', type=str, default='test')
    parser.add_argument('--data_mode', type=str, default='wb')
    parser.add_argument('--save_dir', type=str, default='results_tmp')
    
    args = parser.parse_args()

    main(args)
