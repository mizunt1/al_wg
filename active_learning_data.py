import sys

import torch
from collections import defaultdict
from torch.utils.data import DataLoader
import numpy as np
import torch

class ActiveLearningDataGroups():
    def __init__(self, datasets, dataset_test, num_workers=4, batch_size=64):
        if isinstance(datasets, list):
            self.dataset_list = datasets
            self.dataset = torch.utils.data.ConcatDataset(datasets)
            self.dataset_test = torch.utils.data.ConcatDataset(dataset_test)
            self._create_group_indices()
        else:
            self.dataset = datasets
            self.dataset_test = dataset_test
        self.pool_mask = np.full((len(self.dataset)), True)
        self.train_mask = np.full((len(self.dataset)), False)
        self.pool = torch.utils.data.Subset(self.dataset, None)
        self.train = torch.utils.data.Subset(self.dataset, None)
        self.pool_scores = None
        self.batching = False
        self.num_workers = num_workers
        self._update_indices()
        self.batch_size = batch_size

    def _update_indices(self):
        self.pool.indices = np.nonzero(self.pool_mask)[0]
        self.train.indices = np.nonzero(self.train_mask)[0]
        
    def _create_group_indices(self):
        start_idxs = np.cumsum([len(data) for data in self.dataset_list])
        start_idxs[-1] = start_idxs[-1]-1
        start_idxs = np.insert(start_idxs,0,0)
        self.group_idx = start_idxs
    
    def get_dataloader_all(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def get_dataset_indices(self):
        indices = self.train.indices
        return indices
    
    def create_dataloader_with_indices(self, indices, batch_size=None):
        if batch_size == None:
            batch_size = len(indices)
        dataset_subset = torch.utils.data.Subset(self.dataset, indices)
        return DataLoader(dataset_subset, batch_size=batch_size,
                          num_workers=self.num_workers, pin_memory=True)

        
    def acquire_with_indices(self, indices):
        num_avail_pool = np.sum(self.pool_mask==True)
        assert 0 < len(indices) <= num_avail_pool
        self.pool_mask[indices] = False
        self.train_mask[indices] = True
        self._update_indices()        
    
    def get_indices_one_group(self, group=0, size=10):
        # pool indices from that group
        indexes_for_group = [i for i in range(self.group_idx[group], self.group_idx[group+1])]
        indexes_in_pool = self.pool.indices
        available_from_group = list(set(indexes_for_group).intersection(set(indexes_in_pool)))
        assert size <= len(available_from_group)
        if size!= -1:
            available_indices = np.random.permutation(available_from_group)[:size]
        else:
            available_indices = np.random.permutation(available_from_group)
        return available_indices

    def get_indices_groups(self, group_dict={0:0,1:10}):
        indices = []
        for group, size in group_dict.items():
            indx = self.get_indices_one_group(group=group, size=size)
            indices.extend(indx)
        return indices

    def get_random_available_indices(self, size=10):
        assert size <= len(self.pool)
        available_indices = np.random.permutation(self.pool.indices)[:size]
        return available_indices
        
    def get_pool_loader(self, batch_size, shuffle=False):
        return DataLoader(self.pool, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)
        
    def get_train_and_test_loader(self, batch_size):
        return (DataLoader(self.train, batch_size=self.batch_size,
                           num_workers=self.num_workers, pin_memory=True, shuffle=True),
                DataLoader(self.dataset_test, batch_size=batch_size, shuffle=True))
