from abc import ABC, abstractmethod
from epig import batch_epig
from cmnist_ram import return_log_probs
import random
from tools import calc_ent_per_source_batched, test_batched_per_source, calc_ent_per_point_batched
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
    
class UniformSources(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, source_proportions_in=None, al_size=None):
        self.al_data = al_data
        self.source_proportions_in = source_proportions_in
        self.al_size = al_size
    
    def information_for_acquisition(self, model):
        return None

    def return_indices(self):
        self.source_proportions = spread_remainder(self.al_size, self.source_proportions_in)
        #self.source_proportions = {0:0, 1:self.al_size, 2:0, 3:0, 4:0}
        return self.al_data.get_indices_sources(self.source_proportions)
    
def spread_remainder(al_size, source_proportions):
    source_proportions = source_proportions.copy()
    total_size = sum(source_proportions.values())
    num_sources = len(source_proportions)
    rand_int = randint(0, num_sources -1)
    if total_size == al_size:
        return source_proportions
    if total_size > al_size:
        for i in range(total_size - al_size):
            source = [*source_proportions.keys()][rand_int]
            source_proportions[source] = source_proportions[source] -1
    if total_size < al_size:
        for i in range(al_size - total_size):
            source = [*source_proportions.keys()][rand_int]
            source_proportions[source] = source_proportions[source] + 1
    assert sum(source_proportions.values()) == al_size
    
    return source_proportions

class EntropyPerSource(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None, within_source_acquisition='random',
                 softmax=True, temperature=0.1,
                 num_sources=None, mi=False):
        self.al_data = al_data
        self.pool_indices = self.al_data.pool.indices
        self.al_size = al_size
        self.within_source_acquisition = within_source_acquisition
        self.source_proportions = None
        self.source_ents = None
        self.softmax = softmax
        self.temperature = temperature
        self.num_sources = num_sources
        self.mi = mi
        self.ents = None
        self.source_ids = None
        

    def _calc_source_proportions(self, model, dataloader, al_size):
        self.source_ents, self.ents, self.source_ids = calc_ent_per_source_batched(
            model, dataloader, self.num_sources, mi=self.mi, return_averaged_only=False)
        if self.softmax:
            calculated_prob = softmax(np.array([*self.source_ents.values()])/self.temperature)
            self.source_proportions = {key: round(value*al_size) for key, value in enumerate(calculated_prob)}
            to_log = self.source_ents
        else:
            total_ent = sum(self.source_ents.values())
            self.source_proportions = {key: round((value/total_ent)*al_size) for key, value in self.source_ents.items()}
            to_log = self.source_ents
        self.source_proportions = spread_remainder(al_size, self.source_proportions)
        return to_log
    
    def information_for_acquisition(self, model):        
        to_log = self._calc_source_proportions(
            model, self.al_data.get_pool_loader(), self.al_size)
        return to_log

    def return_indices(self):
        if self.within_source_acquisition == 'random':
            return self.al_data.get_indices_sources(self.source_proportions)
        else:
            final_indices = []
            for source_id, num_points_per_s in self.source_proportions.items():
                ents_one_source = [i for (i,j) in zip(self.ents, self.source_ids) if j == source_id]
                indices_one_source = [i for (i, j) in zip(self.al_data.pool.indices, self.source_ids) if j == source_id]
                assert len(ents_one_source) == len(indices_one_source)
                candidate_batch = get_stochastic_samples(
                    torch.Tensor(ents_one_source), coldness=1, batch_size=num_points_per_s,
                    mode=self.within_source_acquisition)
                location_of_top_points = [indices_one_source[i] for i in candidate_batch.indices]
                final_indices.extend(location_of_top_points)

            return final_indices

class EntropyPerSourceNLargest(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None, n=1, num_sources=None, within_source_acquisition='random'):
        self.al_data = al_data
        self.al_size = al_size
        self.source_proportions = None
        self.n = n
        self.num_sources = num_sources
        self.mi = False
        self.within_source_acquisition = within_source_acquisition
        
    def _largest_ent_source(self, model, dataloader, al_size):
        self.source_ents, self.ents, self.source_ids = calc_ent_per_source_batched(
            model, dataloader, self.num_sources, mi=self.mi, return_averaged_only=False)

        max_sources0 = sorted(self.source_ents.items(), key=lambda item: item[1])[-self.n:]
        max_sources = [item[0] for item in max_sources0]
        source_prop = {key:0 for key, items in self.source_ents.items()}
        sample_per_source = round(al_size /self.n)
        for source_ in max_sources:
            source_prop[source_] = sample_per_source
        return source_prop, self.source_ents

    def return_indices(self):
        #self.source_proportions = {0:0, 1:self.al_size, 2:0, 3:0, 4:0}
        if self.within_source_acquisition == 'random':
            return self.al_data.get_indices_sources(self.source_proportions)
        else:
            final_indices = []
            for source_id, num_points_per_s in self.source_proportions.items():
                ents_one_source = [i for (i,j) in zip(self.ents, self.source_ids) if j == source_id]
                indices_one_source = [i for (i, j) in zip(self.al_data.pool.indices, self.source_ids) if j == source_id]
                assert len(ents_one_source) == len(indices_one_source)
                candidate_batch = get_stochastic_samples(
                    torch.Tensor(ents_one_source), coldness=1, batch_size=num_points_per_s,
                    mode=self.within_source_acquisition)
                location_of_top_points = [indices_one_source[i] for i in candidate_batch.indices]
                final_indices.extend(location_of_top_points)

            return final_indices

    def information_for_acquisition(self, model):
        source_proportions, source_ents = self._largest_ent_source(
            model, self.al_data.get_pool_loader(), self.al_size)
        self.source_proportions = source_proportions
        return source_ents
    

class AccuracyPerSource(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None):
        self.al_data = al_data
        self.al_size = al_size
        self.source_proportions = None
        
    def _acc_per_source_inverse(self, model, data_to_test, num_sources):
        dataloader_mini_test = self.al_data.create_dataloader_with_indices(data_to_test)
        source_accs = test_batched_per_source(model, dataloader_mini_test, num_sources)
        accs = {key: 1/ (value + 1e-2) for key, value in source_accs.items()}
        total_values = sum(accs.values()) 
        source_amounts = {key: round((value/total_values)*self.al_size) for key, value in accs.items()}
        return source_amounts

    def information_for_acquisition(self, model, data_to_test, num_sources, k=3):
        self.source_proportions = self._acc_per_source_inverse(model, data_to_test, num_sources)
        
    def return_indices(self):
        return self.al_data.get_indices_sources(self.source_proportions)

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


class EntropyPerSourceOrdered(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None):
        self.al_data = al_data
        self.al_size = al_size
        self.source_proportions = None

    def _ent_per_source_inverse(self, model, dataloader, num_sources, al_size):
        source_ents = calc_ent_per_source_batched(model, dataloader, num_sources)
        total_ent = sum(source_ents.values())
        normalised_ents = {key: int((value/total_ent)*al_size) for key, value in source_ents.items()}
        return normalised_ents

    def information_for_acquisition(self, model, num_sources):
        source_proportions = self._ent_per_source_inverse(
            model, self.al_data.get_pool_loader(64), num_sources, self.al_size)
        self.source_proportions = source_proportions
        self.indices_selected = []
        for source_id, proportion in source_proportions.items():
            # get all indexes for one source
            indices_all_source = self.al_data.get_indices_one_source(source_id)
            loader = self.al_data.create_dataloader_with_indices(indices_all_source, batch_size=64)
            entropies = calc_ent_per_point_batched(model, loader)
            sorted_by_ent = sorted(zip(entropies, indices_all_source), reverse=True)
            greatest_ent_points = [item[1] for item in sorted_by_ent[:proportion]]
            self.indices_selected.extend(greatest_ent_points)
        return source_proportions
    
    def return_indices(self):
        return self.indices_selected

class EntropyUniformSources(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None,
                 source_proportions=None):
        self.al_data = al_data
        self.al_size = al_size
        self.entropies = None
        self.indices = None
        self.source_proportions = source_proportions
        
    def information_for_acquisition(self, model):
        # calculate source sampling proportions
        pool_loader = self.al_data.get_pool_loader(64)
        self.indices = self.al_data.pool.indices
        self.entropies = calc_ent_per_point_batched(model, pool_loader)
        return self.entropies

    def return_indices(self):
        final_indices = []
        sorted_by_ent = sorted(zip(self.entropies, self.indices), reverse=True)
        for source_id, num_points_per_s in self.source_proportions.items():
            sorted_by_ent_one_source = [ind for ent, ind in sorted_by_ent if ind < self.al_data.source_idx[source_id+1] and ind > self.al_data.source_idx[source_id]][:num_points_per_s]
            final_indices.extend(sorted_by_ent_one_source)
        return final_indices


class EPIG_largest_entropy_source(ActiveLearningAcquisitions):
    def __init__(self, al_data=None, al_size=None,
                 num_sources=None, num_samples_target=200):
        self.al_data = al_data
        self.al_size = al_size
        self.num_samples_target = num_samples_target
        self.num_sources = num_sources
        self.scores_indices = None


    def information_for_acquisition(self, model):
        source_ents = calc_ent_per_source_batched(model,
                                                self.al_data.get_pool_loader(64), self.num_sources)
        # calculate source with maximum entropy
        source_max_ent = max(source_ents, key=source_ents.get)
        # get data for one source
        print(source_max_ent)
        indices_target = self.al_data.get_indices_one_source(source=source_max_ent, size=-1)
        # that would be the target source
        # subsample target
        sampled_target = np.random.choice(indices_target, size=self.num_samples_target, replace=False)
        target_dataloader = self.al_data.create_dataloader_with_indices(sampled_target, batch_size=64)
        # get log probs for target source and whole poolset
        # get epig scores for poolset.         
        log_probs_target = return_log_probs(model, target_dataloader)
        log_probs_pool = return_log_probs(model, self.al_data.get_pool_loader(64))
        epig_scores = batch_epig(log_probs_pool, log_probs_target)
        self.scores_indices = zip(epig_scores, self.al_data.pool.indices) 
        return source_ents
    
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

