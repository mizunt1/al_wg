from data_loading import waterbirds
import wandb
from tools import calc_ent_per_point_batched
from tools import slurm_infos
import numpy as np
import torch
import collections
from waterbirds_dataset import train_batched
import models
import pandas as pd
import os

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
    datasize = [50, 100, 150, 200, 400, 600, 800, 1000, 1250, 1500, 1750, 2000]
    #datasize = [100]
    for size in datasize:
        num_minority_points = int(size*args.minority_prop)
        num_majority_points = size-num_minority_points
        print('size '+ str(size))
        if args.data_mode == 'wb':
            # note that minority points is the number of wl + lw.
            # majority points is the number of ww + ll.

            try:
                training_loader, test_loader, training_data_dict = waterbirds(num_minority_points,
                                                                              num_majority_points,
                                                                              batch_size=args.batch_size,
                                                                              metadata_path='metadata_larger.csv',
                                                                              root_dir='/network/scratch/m/mizu.nishikawa-toomey/waterbird_larger')
            except:
                continue
            # train model for different data amounts and log train and test
            model = getattr(models, args.model_name)
            model = model(2)
            train_acc, test_acc, group_dict, wga = train_batched(
                model, num_epochs=args.num_epochs, lr=args.lr, num_groups=4,
                dataloader=training_loader, dataloader_test=test_loader['val'])
        
            # calculate some kind of UQ metric 
            # save results
            ww_ent = calc_ent_per_point_batched(model, test_loader['ww_test'], mean=True)
            wl_ent = calc_ent_per_point_batched(model, test_loader['wl_test'], mean=True)
            ll_ent = calc_ent_per_point_batched(model, test_loader['ll_test'], mean=True)
            lw_ent = calc_ent_per_point_batched(model, test_loader['lw_test'], mean=True)
            to_log = {'data size': size, 'minority prop': args.minority_prop,
                      'ww_ent': ww_ent, 'wl_ent': wl_ent, 'll_ent': ll_ent, 'lw_ent': lw_ent,
                      'train_acc': train_acc,
                      'wga': wga}
            to_log.update(test_acc)
            results = pd.concat([pd.DataFrame(to_log, index=[0]), results],ignore_index=True)
            wandb.log(to_log)
    dir_name = f"{args.save_dir}/{args.seed}"
    os.makedirs(dir_name, exist_ok=True) 
    results.to_csv(f"{dir_name}/{args.minority_prop}.csv")
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--minority_prop', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--model_name', type=str, default='BayesianNetRes50ULarger')
    parser.add_argument('--project_name', type=str, default='uq_test_wb_exp1')
    parser.add_argument('--data_mode', type=str, default='wb')
    parser.add_argument('--save_dir', type=str, default='results_seeded')
    
    args = parser.parse_args()

    main(args)
