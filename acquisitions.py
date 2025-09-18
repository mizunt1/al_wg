from abc import ABC, abstractmethod
from epig import batch_epig
from cmnist_ram import return_log_probs
import random
from tools import calc_ent_per_group_batched, test_batched_per_group, calc_ent_per_point_batched
import numpy as np

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
    
    def information_for_acquisition(self):
        pass

    def return_indices(self):
        return self.al_data.get_random_available_indices(self.al_size)
    
class UniformGroups(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, group_proportions=None):
        self.al_data = al_data
        self.group_proportions = group_proportions
    
    def information_for_acquisition(self):
        pass

    def return_indices(self):
        return self.al_data.get_indices_groups(self.group_proportions)
    
class EntropyPerGroup(ActiveLearningAcquisitions):
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

    def return_indices(self):
        return self.al_data.get_indices_groups(self.group_proportions)
        
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
        group_amounts = {key: int((value/total_values)*self.al_size) for key, value in accs.items()}
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

    def return_indices(self):
        sorted_by_ent = sorted(zip(self.entropies, self.indices), reverse=True)
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
        
    def information_for_acquisition(self, model, num_groups):
        # calculate group sampling proportions
        pool_loader = self.al_data.get_pool_loader(64)
        self.indices = self.al_data.pool.indices
        self.entropies = calc_ent_per_point_batched(model, pool_loader)
        
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

    def return_indices(self):
        sorted_by_score = sorted(self.scores_indices, reverse=True)
        greatest_ent_points = [item[1] for item in sorted_by_score[:self.al_size]]
        return greatest_ent_points
    

    
