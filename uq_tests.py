from data_loading import waterbirds, celeba_load, celeba_non_sp_load
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
    use_cuda = True
    log = collections.defaultdict(list)
    # load data
    # need to loop through datasize

    #datasize = [50, 100, 150, 200, 400, 600, 800, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000]
    if args.data_mode == 'celeba':
        datasize = [50, 100, 150, 200, 250, 300, 450, 500, 600, 700, 900, 1200, 1400, 2000]
    elif args.data_mode == 'wb':
        datasize = [50, 100, 150, 200, 250, 300, 450, 500, 600, 700, 900, 1200, 1400, 2000]
    elif args.data_mode == 'celeba_non_sp':
        datasize = [50, 100, 150, 200, 250, 300, 450, 500, 600, 700, 900, 1200, 1400, 2000]
    else:
        pass
    for size in datasize:
        num_minority_points = int(size*args.minority_prop)
        num_majority_points = size-num_minority_points
        print('size '+ str(size))
        model = getattr(models, args.model_name)
        model = model(2, args.pretrained, args.frozen_weights)
        if 'clip' in args.model_name.lower():
            model = model.float()
            img_size = 244
            trans_celeba = transforms.Compose([transforms.PILToTensor(), transforms.Resize((img_size,img_size))])
        else:
            img_size = None
            trans_celeba = transforms.Compose([transforms.PILToTensor()])
        if args.mc_drop_p != None:
            set_dropout_p(model, args.mc_drop_p)
            
        if args.data_mode == 'wb':
            # note that minority points is the number of wl + lw.
            # majority points is the number of ww + ll.
            training_loader, test_loader, training_data_dict, test_data_dict = waterbirds(num_minority_points,
                                                                                          num_majority_points,
                                                                                          batch_size=args.batch_size,
                                                                                          metadata_path='metadata_larger.csv',
                                                                                          root_dir='/network/scratch/m/mizu.nishikawa-toomey/waterbird_larger', img_size=img_size)
            #except:
            #    continue
            waterbirds_dummy = WaterbirdsDataset()
            # train model for different data amounts and log train and test

            train_acc, test_acc, group_dict, wga = train_batched(
                model, num_epochs=args.num_epochs, lr=args.lr, num_groups=4,
                dataloader=training_loader, dataloader_test=test_loader, group_mapping_fn=waterbirds_dummy.group_mapping_fn,
                group_string_map=waterbirds_dummy.group_string_map, group_key='metadata')
            # calculate some kind of UQ metric 
            # save results
            root_dir = '/network/scratch/m/mizu.nishikawa-toomey'
            blond_male = CelebA(root_dir, download=True, transform=trans_celeba, split='train_bm')
            celeba = torch.utils.data.DataLoader(blond_male, batch_size=args.batch_size)

            ww_ent = calc_ent_per_point_batched(model, test_data_dict['ww_test'], mean=True)
            wl_ent = calc_ent_per_point_batched(model, test_data_dict['wl_test'], mean=True)
            ll_ent = calc_ent_per_point_batched(model, test_data_dict['ll_test'], mean=True)
            lw_ent = calc_ent_per_point_batched(model, test_data_dict['lw_test'], mean=True)
            celeba = calc_ent_per_point_batched(model, celeba, mean=True)
            to_log = {'data size': size, 'minority prop': args.minority_prop,
                      'ww_ent': ww_ent, 'wl_ent': wl_ent, 'll_ent': ll_ent, 'lw_ent': lw_ent, 'celeba ent': celeba,
                      'train_acc': train_acc,
                      'wga': wga}
            to_log.update(test_acc)
            results = pd.concat([pd.DataFrame(to_log, index=[0]), results],ignore_index=True)
            wandb.log(to_log)
        if args.data_mode == 'celeba':
            training_loader, test_loader, training_data_dict, test_data_dict = celeba_load(num_minority_points,
                                                                                           num_majority_points, img_size=img_size,
                                                                                           batch_size=args.batch_size)
            root_dir = '/network/scratch/m/mizu.nishikawa-toomey'
            celeba_dummy = CelebA(root_dir, download=True, transform=trans_celeba, split='train_bm')
            training_loaderwb, test_loaderwb, _, _ = waterbirds(1000,
                                                                1000,
                                                                batch_size=args.batch_size, img_size=img_size,
                                                                metadata_path='metadata_larger.csv',
                                                                root_dir='/network/scratch/m/mizu.nishikawa-toomey/waterbird_larger')

            waterbirds_dummy = WaterbirdsDataset()
            train_acc, test_acc, group_dict, wga = train_batched(
                model, num_epochs=args.num_epochs, lr=args.lr, num_groups=4,
                dataloader=training_loader, dataloader_test=test_loader, group_mapping_fn=celeba_dummy.group_mapping_fn,
                group_string_map=celeba_dummy.group_string_map, group_key='metadata')
            mb_ent = calc_ent_per_point_batched(model, DataLoader(test_data_dict['mb_test'], batch_size=args.batch_size),
                                                mean=True)
            fb_ent = calc_ent_per_point_batched(model, DataLoader(test_data_dict['fb_test'], batch_size=args.batch_size),
                                                mean=True)
            mnb_ent = calc_ent_per_point_batched(model, DataLoader(test_data_dict['mnb_test'], batch_size=args.batch_size),
                                                 mean=True)
            fnb_ent = calc_ent_per_point_batched(model, DataLoader(test_data_dict['fnb_test'], batch_size=args.batch_size),
                                                 mean=True)
            wb = calc_ent_per_point_batched(model, test_loaderwb, mean=True)

            to_log = {'data size': size, 'minority prop': args.minority_prop,
                      'mb_ent': mb_ent, 'fb_ent': fb_ent, 'mnb_ent': mnb_ent, 'fnb_ent': fnb_ent, 'wb ent': wb,
                      'train_acc': train_acc,
                      'wga': wga}
            to_log.update(test_acc)
            results = pd.concat([pd.DataFrame(to_log, index=[0]), results],ignore_index=True)
            wandb.log(to_log)

            dir_name = f"{args.data_mode}/{args.save_dir}/{args.seed}"
            os.makedirs(dir_name, exist_ok=True) 
            results.to_csv(f"{dir_name}/{args.minority_prop}.csv")

        if args.data_mode == 'celeba_non_sp':
            training_loader, test_loader, training_data_dict, test_data_dict = celeba_non_sp_load(num_minority_points,
                                                                                                  num_majority_points,
                                                                                                  batch_size=args.batch_size,
                                                                                                  img_size=img_size)
            root_dir = '/network/scratch/m/mizu.nishikawa-toomey'
            celeba_dummy = CelebA(root_dir, download=True, transform=trans_celeba, split='train_bm')
            training_loaderwb, test_loaderwb, _, _ = waterbirds(1000,
                                                                1000,
                                                                batch_size=args.batch_size,
                                                                metadata_path='metadata_larger.csv',
                                                                root_dir='/network/scratch/m/mizu.nishikawa-toomey/waterbird_larger', img_size=img_size)

            waterbirds_dummy = WaterbirdsDataset()
            train_acc, test_acc, group_dict, wga = train_batched(
                model, num_epochs=args.num_epochs, lr=args.lr, num_groups=4,
                dataloader=training_loader, dataloader_test=test_loader, group_mapping_fn=celeba_dummy.group_mapping_fn,
                group_string_map=celeba_dummy.group_string_map, group_key='metadata')
            mb_ent = calc_ent_per_point_batched(model, DataLoader(test_data_dict['mb_test'], batch_size=args.batch_size),
                                                mean=True)
            fb_ent = calc_ent_per_point_batched(model, DataLoader(test_data_dict['fb_test'], batch_size=args.batch_size),
                                                mean=True)
            mnb_ent = calc_ent_per_point_batched(model, DataLoader(test_data_dict['mnb_test'], batch_size=args.batch_size),
                                                 mean=True)
            fnb_ent = calc_ent_per_point_batched(model, DataLoader(test_data_dict['fnb_test'], batch_size=args.batch_size),
                                                 mean=True)
            wb = calc_ent_per_point_batched(model, test_loaderwb, mean=True)

            to_log = {'data size': size, 'minority prop': args.minority_prop,
                      'mb_ent': mb_ent, 'fb_ent': fb_ent, 'mnb_ent': mnb_ent, 'fnb_ent': fnb_ent, 'wb ent': wb,
                      'train_acc': train_acc,
                      'wga': wga}
            to_log.update(test_acc)
            results = pd.concat([pd.DataFrame(to_log, index=[0]), results],ignore_index=True)
            wandb.log(to_log)

            dir_name = f"{args.data_mode}/{args.save_dir}/{args.seed}"
            os.makedirs(dir_name, exist_ok=True) 
            results.to_csv(f"{dir_name}/{args.minority_prop}.csv")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--frozen_weights', default=False, action='store_true')
    parser.add_argument('--pretrained', default=True, action='store_false')
    parser.add_argument('--minority_prop', type=float, default=0.2)
    parser.add_argument('--mc_drop_p', type=float, default=None)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--model_name', type=str, default='BayesianNetRes50ULarger')
    parser.add_argument('--project_name', type=str, default='uq_test_wb_fin')
    parser.add_argument('--data_mode', type=str, default='wb')
    parser.add_argument('--save_dir', type=str, default='results_tmp')
    
    args = parser.parse_args()

    main(args)
