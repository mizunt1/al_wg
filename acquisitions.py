from abc import ABC, abstractmethod
from tools import calc_ent_per_group_batched, test_batched_per_group, calc_ent_per_point_batched

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
        self.indexes = None

    def information_for_acquisition(self, model):
        pool_loader = self.al_data.get_pool_loader(64)
        self.indexes = self.al_data.pool.indices
        self.entropies = calc_ent_per_point_batched(model, pool_loader)

    def return_indices(self):
        sorted_by_ent = sorted(zip(self.entropies, self.indexes), reverse=True)
        greatest_ent_points = [item[1] for item in sorted_by_ent[:self.al_size]]
        return greatest_ent_points
