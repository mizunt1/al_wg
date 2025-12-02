from abc import ABC, abstractmethod
from epig import batch_epig
from cmnist_ram import return_log_probs
import random
from tools import calc_ent_per_group_batched, test_batched_per_group, calc_ent_per_point_batched
import numpy as np
from scipy.special import softmax
from random import randint
from stochastic_acquisition import get_stochastic_samples
import torch

class ActiveLearningAcquisitions(ABC):
    @abstractmethod
    def return_indices(self):
        pass
    @abstractmethod
    def information_for_acquisition(self):
        return None

class Random(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None):
        self.al_data = al_data
        self.al_size = al_size
    
    def information_for_acquisition(self, model):
        return None

    def return_indices(self):
        return self.al_data.get_random_available_indices(self.al_size)
    
class UniformGroups(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, group_proportions_in=None, al_size=None):
        self.al_data = al_data
        self.group_proportions_in = group_proportions_in
        self.al_size = al_size
    
    def information_for_acquisition(self, model):
        return None

    def return_indices(self):
        self.group_proportions = spread_remainder(self.al_size, self.group_proportions_in)
        return self.al_data.get_indices_groups(self.group_proportions)
    
def spread_remainder(al_size, group_proportions):
    group_proportions = group_proportions.copy()
    total_size = sum(group_proportions.values())
    num_groups = len(group_proportions)
    rand_int = randint(0, num_groups -1)
    if total_size == al_size:
        return group_proportions
    if total_size > al_size:
        for i in range(total_size - al_size):
            group = [*group_proportions.keys()][rand_int]
            group_proportions[group] = group_proportions[group] -1
    if total_size < al_size:
        for i in range(al_size - total_size):
            group = [*group_proportions.keys()][rand_int]
            group_proportions[group] = group_proportions[group] + 1
    assert sum(group_proportions.values()) == al_size
    
    return group_proportions

class EntropyPerGroup(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None, within_group_acquisition='random',
                 softmax=True, temperature=0.1,
                 num_groups=None, mi=False):
        self.al_data = al_data
        self.pool_indices = self.al_data.pool.indices
        self.al_size = al_size
        self.within_group_acquisition = within_group_acquisition
        self.group_proportions = None
        self.group_ents = None
        self.softmax = softmax
        self.temperature = temperature
        self.num_groups = num_groups
        self.mi = mi
        self.ents = None
        self.group_ids = None
        

    def _calc_group_proportions(self, model, dataloader, al_size):
        self.group_ents, self.ents, self.group_ids = calc_ent_per_group_batched(
            model, dataloader, self.num_groups, mi=self.mi, return_averaged_only=False)
        if self.softmax:
            calculated_prob = softmax(np.array([*self.group_ents.values()])/self.temperature)
            self.group_proportions = {key: round(value*al_size) for key, value in enumerate(calculated_prob)}
            to_log = self.group_ents
        else:
            total_ent = sum(self.group_ents.values())
            self.group_proportions = {key: round((value/total_ent)*al_size) for key, value in self.group_ents.items()}
            to_log = self.group_ents
        self.group_proportions = spread_remainder(al_size, self.group_proportions)
        return to_log
    
    def information_for_acquisition(self, model):        
        to_log = self._calc_group_proportions(
            model, self.al_data.get_pool_loader(), self.al_size)
        return to_log

    def return_indices(self):
        if self.within_group_acquisition == 'random':
            return self.al_data.get_indices_groups(self.group_proportions)
        else:
            final_indices = []
            for group_id, num_points_per_g in self.group_proportions.items():
                ents_one_group = [i for (i,j) in zip(self.ents, self.group_ids) if j == group_id]
                indices_one_group = [i for (i, j) in zip(self.al_data.pool.indices, self.group_ids) if j == group_id]
                assert len(ents_one_group) == len(indices_one_group)
                candidate_batch = get_stochastic_samples(
                    torch.Tensor(ents_one_group), coldness=1, batch_size=num_points_per_g,
                    mode=self.within_group_acquisition)
                location_of_top_points = [indices_one_group[i] for i in candidate_batch.indices]
                final_indices.extend(location_of_top_points)

            return final_indices

class EntropyPerGroupNLargest(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None, n=1, num_groups=None, within_group_acquisition='random'):
        self.al_data = al_data
        self.al_size = al_size
        self.group_proportions = None
        self.n = n
        self.num_groups = num_groups
        self.mi = False
        self.within_group_acquisition = within_group_acquisition
        
    def _largest_ent_group(self, model, dataloader, al_size):
        self.group_ents, self.ents, self.group_ids = calc_ent_per_group_batched(
            model, dataloader, self.num_groups, mi=self.mi, return_averaged_only=False)

        max_groups0 = sorted(self.group_ents.items(), key=lambda item: item[1])[-self.n:]
        max_groups = [item[0] for item in max_groups0]
        group_prop = {key:0 for key, items in group_ents.items()}
        sample_per_group = round(al_size /self.n)
        for group_ in max_groups:
            group_prop[group_] = sample_per_group
        return group_prop, group_ents

    def return_indices(self):
        if self.within_group_acquisition == 'random':
            return self.al_data.get_indices_groups(self.group_proportions)
        else:
            final_indices = []
            for group_id, num_points_per_g in self.group_proportions.items():
                ents_one_group = [i for (i,j) in zip(self.ents, self.group_ids) if j == group_id]
                indices_one_group = [i for (i, j) in zip(self.al_data.pool.indices, self.group_ids) if j == group_id]
                assert len(ents_one_group) == len(indices_one_group)
                candidate_batch = get_stochastic_samples(
                    torch.Tensor(ents_one_group), coldness=1, batch_size=num_points_per_g,
                    mode=self.within_group_acquisition)
                location_of_top_points = [indices_one_group[i] for i in candidate_batch.indices]
                final_indices.extend(location_of_top_points)

            return final_indices

    def information_for_acquisition(self, model):
        group_proportions, group_ents = self._largest_ent_group(
            model, self.al_data.get_pool_loader(), self.al_size)
        self.group_proportions = group_proportions
        return group_ents
    

class AccuracyPerGroup(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None):
        self.al_data = al_data
        self.al_size = al_size
        self.group_proportions = None
        
    def _acc_per_group_inverse(self, model, data_to_test, num_groups):
        dataloader_mini_test = self.al_data.create_dataloader_with_indices(data_to_test)
        group_accs = test_batched_per_group(model, dataloader_mini_test, num_groups)
        accs = {key: 1/ (value + 1e-2) for key, value in group_accs.items()}
        total_values = sum(accs.values()) 
        group_amounts = {key: round((value/total_values)*self.al_size) for key, value in accs.items()}
        return group_amounts

    def information_for_acquisition(self, model, data_to_test, num_groups, k=3):
        self.group_proportions = self._acc_per_group_inverse(model, data_to_test, num_groups)
        
    def return_indices(self):
        return self.al_data.get_indices_groups(self.group_proportions)

    def return_indices_random(self):
        return self.al_data.get_random_available_indices(self.al_size)

class Entropy(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None):
        self.al_data = al_data
        self.al_size = al_size
        self.entropies = None
        self.indices = None

    def information_for_acquisition(self, model):
        pool_loader = self.al_data.get_pool_loader(64)
        self.indices = self.al_data.pool.indices
        self.entropies = calc_ent_per_point_batched(model, pool_loader)
        return None
        
    def return_indices(self):
        sorted_by_ent = sorted(zip(self.entropies, self.indices), reverse=True)
        greatest_ent_points = [item[1] for item in sorted_by_ent[:self.al_size]]
        return greatest_ent_points

class MI(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None):
        self.al_data = al_data
        self.al_size = al_size
        self.mi = None
        self.indices = None

    def information_for_acquisition(self, model):
        pool_loader = self.al_data.get_pool_loader(64)
        self.indices = self.al_data.pool.indices
        self.mi = calc_ent_per_point_batched(model, pool_loader, mi=True)
        return None
    
    def return_indices(self):
        sorted_by_ent = sorted(zip(self.mi, self.indices), reverse=True)
        greatest_ent_points = [item[1] for item in sorted_by_ent[:self.al_size]]
        return greatest_ent_points


class EntropyPerGroupOrdered(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None):
        self.al_data = al_data
        self.al_size = al_size
        self.group_proportions = None

    def _ent_per_group_inverse(self, model, dataloader, num_groups, al_size):
        group_ents = calc_ent_per_group_batched(model, dataloader, num_groups)
        total_ent = sum(group_ents.values())
        normalised_ents = {key: int((value/total_ent)*al_size) for key, value in group_ents.items()}
        return normalised_ents

    def information_for_acquisition(self, model, num_groups):
        group_proportions = self._ent_per_group_inverse(
            model, self.al_data.get_pool_loader(64), num_groups, self.al_size)
        self.group_proportions = group_proportions
        self.indices_selected = []
        for group_id, proportion in group_proportions.items():
            # get all indexes for one group
            indices_all_group = self.al_data.get_indices_one_group(group_id)
            loader = self.al_data.create_dataloader_with_indices(indices_all_group, batch_size=64)
            entropies = calc_ent_per_point_batched(model, loader)
            sorted_by_ent = sorted(zip(entropies, indices_all_group), reverse=True)
            greatest_ent_points = [item[1] for item in sorted_by_ent[:proportion]]
            self.indices_selected.extend(greatest_ent_points)
        return group_proportions
    
    def return_indices(self):
        return self.indices_selected

class EntropyUniformGroups(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None,
                 group_proportions=None):
        self.al_data = al_data
        self.al_size = al_size
        self.entropies = None
        self.indices = None
        self.group_proportions = group_proportions
        
    def information_for_acquisition(self, model):
        # calculate group sampling proportions
        pool_loader = self.al_data.get_pool_loader(64)
        self.indices = self.al_data.pool.indices
        self.entropies = calc_ent_per_point_batched(model, pool_loader)
        return self.entropies

    def return_indices(self):
        final_indices = []
        sorted_by_ent = sorted(zip(self.entropies, self.indices), reverse=True)
        for group_id, num_points_per_g in self.group_proportions.items():
            sorted_by_ent_one_group = [ind for ent, ind in sorted_by_ent if ind < self.al_data.group_idx[group_id+1] and ind > self.al_data.group_idx[group_id]][:num_points_per_g]
            final_indices.extend(sorted_by_ent_one_group)
        return final_indices


class EPIG_largest_entropy_group(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None,
                 num_groups=None, num_samples_target=200):
        self.al_data = al_data
        self.al_size = al_size
        self.num_samples_target = num_samples_target
        self.num_groups = num_groups
        self.scores_indices = None


    def information_for_acquisition(self, model):
        group_ents = calc_ent_per_group_batched(model,
                                                self.al_data.get_pool_loader(64), self.num_groups)
        # calculate group with maximum entropy
        group_max_ent = max(group_ents, key=group_ents.get)
        # get data for one group
        print(group_max_ent)
        indices_target = self.al_data.get_indices_one_group(group=group_max_ent, size=-1)
        # that would be the target group
        # subsample target
        sampled_target = np.random.choice(indices_target, size=self.num_samples_target, replace=False)
        target_dataloader = self.al_data.create_dataloader_with_indices(sampled_target, batch_size=64)
        # get log probs for target group and whole poolset
        # get epig scores for poolset.         
        log_probs_target = return_log_probs(model, target_dataloader)
        log_probs_pool = return_log_probs(model, self.al_data.get_pool_loader(64))
        epig_scores = batch_epig(log_probs_pool, log_probs_target)
        self.scores_indices = zip(epig_scores, self.al_data.pool.indices) 
        return group_ents
    
    def return_indices(self):
        sorted_by_score = sorted(self.scores_indices, reverse=True)
        greatest_ent_points = [item[1] for item in sorted_by_score[:self.al_size]]
        return greatest_ent_points
    
if __name__ == "__main__":
    for i in range(20):
        new = spread_remainder(30, {1:15, 2:14, 3:0})
    print(new)
    new = spread_remainder(30, {1:15, 2:15, 3:1})
    print(new)
    new = spread_remainder(30, {1:15, 2:15, 3:1})
    print(new)

