import sys

import torch
from collections import defaultdict
from torch.utils.data import DataLoader
import numpy as np
import torch.utils.data as data

class ActiveLearningData():
    def __init__(self, dataset, dataset_test, num_workers=4):
        self.dataset = dataset
        self.dataset_test = dataset_test
        self.pool_mask = np.full((len(dataset)), True)
        self.train_mask = np.full((len(dataset)), False)
        self.pool_groups = None
        self.train_groups = None
        self.pool = data.Subset(dataset, None)
        self.train = data.Subset(dataset, None)
        self._update_indices()
        self.group_counts_train = None
        self.pool_scores = None
        self.batching = False
        self._update_indices()
        self.num_workers = num_workers

    def _update_indices(self):
        self.pool.indices = np.nonzero(self.pool_mask)[0]
        self.train.indices = np.nonzero(self.train_mask)[0]

    def update_group_labels(self, group_preds, data_type):
        if data_type == 'pool':
            self.pool_groups = group_preds
        elif data_type == 'train':
            self.train_groups = group_preds
            self._update_group_counts()
        
        else:
            print('data type not correct')
            sys.exit()

    def get_dataloader_all(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def update_group_all(self, group_hats):
        self.dataset.update_input_groups(group_hats)

    def _update_group_counts(self):
        counts_dict = defaultdict()
        items, counts = np.unique(self.train_groups.detach().cpu().numpy(), return_counts=True)
        for item, count in zip(items, counts):
            counts_dict[str(int(item))] = count
        self.group_counts_train = counts_dict

    def get_dataset_indices(self):
        indices = self.train.indices
        return indices
    
    def acquire(self, indices):
        num_avail_pool = np.sum(self.pool_mask==True)
        assert 0 < len(indices) <= num_avail_pool
        self.pool_mask[indices] = False
        self.train_mask[indices] = True
        self._update_indices()

    def get_random_available_indices(self, size):
        assert 0 <= size <= len(self.pool)
        available_indices = np.random.permutation(self.pool.indices)[:size]
        return available_indices

    def get_pool_loader(self, batch_size):
        return DataLoader(self.pool, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
        
    def get_train_loader(self, batch_size):
        return DataLoader(self.train, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
        
    def len_train(self):
        return len(self.train)

    def len_pool(self):
        return len(self.pool)

    def get_test_loader(self, batch_size):
        return DataLoader(self.dataset_test, batch_size=batch_size, shuffle=True)

    def get_oracle_groups_pool(self):
        return self.dataset.get_oracle_groups(self.pool.indices)
        
    def get_oracle_groups_train(self):
        return self.dataset.get_oracle_groups(self.train.indices)

    def get_oracle_groups_idx(self, idx):
        return self.dataset.get_oracle_groups(idx)

    def get_oracle_train_counts(self):
        groups = self.get_oracle_groups_train()
        counts_dict = defaultdict()
        items, counts = np.unique(groups, return_counts=True)
        for item, count in zip(items, counts):
            counts_dict[str(int(item))] = count
        return counts_dict

    def true_rare_group_in_pred_pool(self, preds):
        """ out of all the points that were predicted group one, which were actually one"""
        oracle_groups_pool = self.get_oracle_groups_pool()
        is_one = sum(np.logical_and(oracle_groups_pool.bool().numpy(), preds.numpy().astype(bool))) / sum(preds)
        return is_one

    def true_common_group_in_pred_pool(self, preds):
        """ out of all the points that were predicted group zero, which were actually one"""
        oracle_groups_pool = self.get_oracle_groups_pool()
        is_zero = sum(np.logical_and(oracle_groups_pool.bool().numpy()==0, preds.numpy().astype(bool)==0)) / sum(preds==0)
        return is_zero

    def true_rare_group_in_pred_train(self, preds):
        oracle_groups_train = self.get_oracle_groups_train()
        is_one = sum(np.logical_and(oracle_groups_train.bool().numpy(), preds.numpy().astype(bool))) / sum(preds)
        return is_one

