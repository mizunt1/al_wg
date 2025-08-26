import itertools
import collections
import os
import numpy as np
import math
import torch
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from scm import scm_aleatoric
from models import erm, Linear, test
from torch.distributions import Categorical 
import matplotlib.pyplot as plt

def provide_groupings_test_acc(matrix, k=0.05):
    """
    using a matrix of test accuracies, group the groups
    that have high test accuracy. 
    """
    len_confusion = matrix.shape[0]
    dict_groupings = collections.defaultdict(list)
    for j in range(len_confusion):
        for i in range(len_confusion):
            if j == i:
                pass
            else:
                in_dist_acc = matrix[j][j]
                out_dist_acc = matrix[i][j]

                if in_dist_acc - out_dist_acc < k:
                    # if difference in acc is small merge
                    dict_groupings[j].append(i)
                else:
                    # if difference in acc is large dont merge
                    dict_groupings[j].append(None)
    sets = []
    for item in dict_groupings:
        if dict_groupings[item] == None:
            sets.append({item})
        created_set = set(dict_groupings[item])
        created_set.add(item)
        sets.append(created_set)

    groupings = []
    for set_item in sets:
        if None in set_item:
            set_item.remove(None)
        len_set = len(set_item)
        is_subset = [item.issubset(set_item) for item in sets]
        if len_set == sum(is_subset) and set_item not in groupings:
            groupings.append(set_item)
    
    flattened = [item for s in groupings for item in s]
    for item in dict_groupings:
        if item not in flattened:
            groupings.append({item})
    return groupings

def provide_group_weights(groupings, datasets):
    total_points = sum([len(data) for data in datasets])
    num_assigned_groups = len(groupings)
    weights = []
    for assigned_group in groupings:
        num_given_groups_in_assigned_group = len(assigned_group)
        for original_group_id in assigned_group:
            current_proportion = len(datasets[original_group_id])/total_points
            desired_proportion = 1/(num_given_groups_in_assigned_group*num_assigned_groups)
            weight = desired_proportion/current_proportion
            weights.append(weight)
    return weights
    

def provide_groupings(model, data, target, k=0.75):
    """
    Function that groups classes and provides new labels for these grouped
    classes. Two classes are grouped as one class (the label being the class)
    with the lower index value, if: the sum of number of correct predictions *k for the
    two classes is less than the sum of the confusion of the two classes. i.e. the sum
    of i being confused for j and j being confused for i. 
    The groups are then merged if this condition is satisfied, and new targets are given
    to reflect the merging of the classes.
    """
    target = torch.clone(target)
    prediction = model(data).argmax(axis=1)
    confusion = confusion_matrix(target, prediction)
    len_confusion = confusion.shape[0]
    matrix_merged = True
    dict_groupings = collections.defaultdict()
    while matrix_merged:
        matrix_merged = False
        for i,j in itertools.combinations(range(len(np.unique(target))), 2):
            total_correct = confusion[i,i] + confusion[j,j]
            exchanged_incorrect = confusion[i,j] + confusion[j,i]
            if k*total_correct < exchanged_incorrect:
                matrix_merged = True
                smaller = min(i,j)
                larger = max(i,j)
                target[target==i] = smaller
                target[target==j] = smaller
                dict_groupings[larger] = smaller
    return dict_groupings
            
def stack_features(data):
    return {'x': np.vstack((data['xsp'], data['xc'])).T.astype('float32'), 'y': data['y']}
    
def query_data(sampling_proportions, data_pool, al_batch_size):
    """
    Given proportion of samples from each u_hat, turn queried masks to 1.
    """
    unqueried_data_idx = data_pool['index'][data_pool['queried'] == 0]
    num_queried = 0
    for group_id, proportion in enumerate(sampling_proportions):
        num_samples = int(proportion*al_batch_size)
        data_for_group_idx = data_pool['index'][data_pool['u_hat'] == group_id]
        group_unqueried_idx = np.intersect1d(unqueried_data_idx, data_for_group_idx)
        to_query = group_unqueried_idx[0:num_samples]
        data_pool['queried'][to_query] = 1
    return data_pool

def return_train_data(data_pool):
    return data_pool['x'][data_pool['queried'] == 1], data_pool['y'][data_pool['queried'] == 1]

def cross_entropy_per_group(model, data_test):
    groups = np.unique(data_test['u_hat'])
    entropy = nn.CrossEntropyLoss(reduction='none')
    output = model(torch.from_numpy(data_test['x']))
    ents = entropy(output, torch.from_numpy(data_test['y'])).detach().numpy()
    group_ents = []
    for group in groups:
        average_group_ent = sum(ents[data_test['u_hat'] == group])/len(data_test['u_hat'] == group)
        group_ents.append(average_group_ent)
    return np.array(group_ents)
 
def cross_entropy(model, x, y, transforms):
    data = transforms(x).squeeze(0)
    out = model(data)    
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(out, y) #+ l1_norm*0.05
    return loss.detach()

def sampling_proportions_entropy(entropy_0, entropy_1):
    """
    Given first entropy values for each group and second entropy values for each group
    return the sampling proportions for each group.
    Each entropy value is given as a dictionary where the key is group id and value is the
    average entropy for that group on the test set.
    """
    if entropy_0[0] == -1:
        sampling_proportions = np.array([1/len(entropy_0) for i in range(len(entropy_0))])
    else:
        entropy_change = entropy_0 - entropy_1
        entropy_norm = abs(entropy_change)/(entropy_1)
        sampling_proportions = entropy_norm/sum(entropy_norm)
        # calculate sampling proportions based on entropy change of each group
    #sampling_proportions = np.array([1/len(entropy_0) for i in range(len(entropy_0))])
    return sampling_proportions

def phase_1(model, data_pool_x, data_pool_label, data_test_x, data_test_label):
    """
    Phase 1 consists of training a group classifier. 
    Then grouping groups based on mistakes to estimate the 
    unobserved intervention u_hat. 
    returns u_hat assignment for each pool point
    """
    # train group classifier.
    # test on test data.
    # define groups u_hat for all pool data based on group classifier mistakes. 
    correct, model = erm(model, data_test_x, data_test_label)
    test_accuracy = test(model, data_pool_x, data_pool_label)
    grouping_labels = provide_groupings(model, data_pool_x, data_pool_label)
    return grouping_labels


def phase_2(model_target, data_pool, data_test, al_batch_size, al_iters):
    """
    Sample from each u_hat and train a target classifier.
    Calculate change in entropy from each group.
    """
    data_pool['queried'] = np.array([0 for i in range(len(data_pool['y']))])
    num_u_hats = len(np.unique(data_pool['u_hat']))
    sampling_proportions = np.array([1/num_u_hats for i in range(num_u_hats)])
    u_hat_entropies_0 = np.array([0 for i in range(num_u_hats)])
    print(sampling_proportions)
    for al_iter in range(al_iters):
        print(f'al iters: {al_iter}')
        data_pool = query_data(sampling_proportions, data_pool, al_batch_size)
        data_train_x, data_train_y = return_train_data(data_pool)
        correct, model_target = erm(model_target,
                             torch.from_numpy(data_train_x),
                             torch.from_numpy(data_train_y))
        test_accuracy = test(model_target,
                             torch.from_numpy(data_test['x']), torch.from_numpy(data_test['y']))
        
        u_hat_entropies_1 = cross_entropy_per_group(model_target, data_test)
        sampling_proportions = sampling_proportions_entropy(u_hat_entropies_0, u_hat_entropies_1)
        print('u hat entropies 0', u_hat_entropies_0)
        print('u hat entropies 1', u_hat_entropies_1)
        print('sampling proportions', sampling_proportions)
        u_hat_entropies_0 = u_hat_entropies_1


def plot_dictionary(data):
    num_plots = 7
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 3 * num_plots), sharex=True)
    
    axes[0].plot(data['ent1'], label='ent1')
    axes[0].plot(data['ent2'], label='ent2')
    axes[0].set_title('ent')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(data['cross_ent_1'], label='xent1')
    axes[1].plot(data['cross_ent_2'], label='xent2')
    axes[1].set_title('xent')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(data['train_acc'], label='train_acc', color='black')
    axes[2].set_title('train acc')
    axes[2].grid(True)

    axes[3].plot(data['test_acc'], label='test_acc', color='black')
    axes[3].set_title('test acc')
    axes[3].grid(True)

    axes[4].plot(data['num points'], label='num points', color='black')
    axes[4].set_title('num_points')
    axes[4].grid(True)

    axes[5].plot(data['sp acc'], label='sp acc', color='black')
    axes[5].set_title('data with sp noise 0, causal noise max. accuracy')
    axes[5].grid(True)

    axes[6].plot(data['causal acc'], label='causal acc', color='black')
    axes[6].set_title('data with causal noise 0, sp noise max. accuracy')
    axes[6].grid(True)

    plt.tight_layout()
    plt.legend()
    plt.show()
    
    
def apply_u_hats(g, mapping):
    return np.asarray([mapping[x] if x in mapping.keys() else x for x in g])

def entropy_drop_out(model, x, transforms=None, num_classes=2, num_models=100):
    if transforms is not None:
        x = transforms(x)
    out = model(x, k=num_models).detach().cpu()
    N, K, C = out.shape
    log_probs_N_K_C = out.to(torch.double)
    mean_log_probs_n_C = torch.logsumexp(log_probs_N_K_C, dim=1) - math.log(K)
    nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)
    nats_n_C[torch.isnan(nats_n_C)] = 0.0
    entropies = -torch.sum(nats_n_C, dim=1)
    return entropies

def cross_entropy(model, x, y):
    out = model(x)
    out = out.squeeze(1)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(out, y)
    return loss.detach()

def calc_ent_batched(model, dataloader, num_models=100):
    total_ent = 0
    total_xent = 0
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()
    model.to(device)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        #data = data.reshape(-1, 3*28*28)
        total_ent += torch.mean(entropy_drop_out(model, data, num_models=num_models))
        total_xent += cross_entropy(model, data, target)
    return total_ent.cpu().item(), total_xent.cpu().item()

def slurm_infos():
    return {
        'slurm/job_id': os.getenv('SLURM_JOB_ID'),
        'slurm/job_user': os.getenv('SLURM_JOB_USER'),
        'slurm/job_partition': os.getenv('SLURM_JOB_PARTITION'),
        'slurm/cpus_per_node': os.getenv('SLURM_JOB_CPUS_PER_NODE'),
        'slurm/num_nodes': os.getenv('SLURM_JOB_NUM_NODES'),
        'slurm/nodelist': os.getenv('SLURM_JOB_NODELIST'),
        'slurm/cluster_name': os.getenv('SLURM_CLUSTER_NAME'),
        'slurm/array_task_id': os.getenv('SLURM_ARRAY_TASK_ID')
    }

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    n_tr = 1000
    n_test = 300
    al_iters = 10
    al_batch_size = 8
    data_pool = scm_aleatoric(n_tr)
    data_test = scm_aleatoric(n_test)
    data_pool_x = torch.from_numpy(data_pool['x'])
    data_pool_label = torch.from_numpy(data_pool['g'])
    data_test_x = torch.from_numpy(data_test['x'])
    data_test_label = torch.from_numpy(data_test['g'])
    model_groups = Linear(2,4)
    new_groupings = phase_1(model_groups, data_pool_x, data_pool_label, data_test_x, data_test_label)
    u_hat = apply_u_hats(data_pool['g'], new_groupings)
    u_hat_test = apply_u_hats(data_test['g'], new_groupings)
    data_pool['u_hat'] = u_hat
    data_test['u_hat'] = u_hat_test
    model_target = Linear(2,3)
    phase_2(model_target, data_pool, data_test, al_batch_size, al_iters)
