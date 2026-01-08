import os
import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import collections
from torch.nn import functional as F

def cross_entropy_per_source(model, data_test):
    sources = np.unique(data_test['u_hat'])
    entropy = nn.CrossEntropyLoss(reduction='none')
    output = model(torch.from_numpy(data_test['x']))
    ents = entropy(output, torch.from_numpy(data_test['y'])).detach().numpy()
    source_ents = []
    for source in sources:
        average_source_ent = sum(ents[data_test['u_hat'] == source])/len(data_test['u_hat'] == source)
        source_ents.append(average_source_ent)
    return np.array(source_ents)

def sampling_proportions_entropy(entropy_0, entropy_1):
    """
    Given first entropy values for each source and second entropy values for each source
    return the sampling proportions for each source.
    Each entropy value is given as a dictionary where the key is source id and value is the
    average entropy for that source on the test set.
    """
    if entropy_0[0] == -1:
        sampling_proportions = np.array([1/len(entropy_0) for i in range(len(entropy_0))])
    else:
        entropy_change = entropy_0 - entropy_1
        entropy_norm = abs(entropy_change)/(entropy_1)
        sampling_proportions = entropy_norm/sum(entropy_norm)
        # calculate sampling proportions based on entropy change of each source
    #sampling_proportions = np.array([1/len(entropy_0) for i in range(len(entropy_0))])
    return sampling_proportions

def plot_dictionary(data):
    num_plots = 9
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

    
    axes[4].plot(data['num points'], label='num points', color='black')
    axes[4].set_title('num_points')
    axes[4].grid(True)

    axes[5].plot(data['sp acc'], label='sp acc', color='black')
    axes[5].set_title('data with sp noise 0, causal noise max. accuracy')
    axes[5].grid(True)

    axes[6].plot(data['causal acc'], label='causal acc', color='black')
    axes[6].set_title('data with causal noise 0, sp noise max. accuracy')
    axes[6].grid(True)

    axes[7].plot(data['g0 points'], color='black')
    axes[7].set_title('num g0 points')
    axes[7].grid(True)

    axes[8].plot(data['g1 points'], color='black')
    axes[8].set_title('num g1 points')
    axes[8].grid(True)


    plt.tight_layout()
    plt.legend()
    plt.show()
    
def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    #https://github.com/BlackHC/active_learning_redux/blob/master/batchbald_redux/joint_entropy.py#L40
    N, K, C = log_probs_N_K_C.shape
    mean_log_probs_n_C = torch.logsumexp(log_probs_N_K_C, dim=1) - math.log(K)
    nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)
    nats_n_C[torch.isnan(nats_n_C)] = 0.0
    entropies = -torch.sum(nats_n_C, dim=1)
    return entropies
    
def entropy_drop_out(model, x, transforms=None, num_classes=2, num_models=100):
    if transforms is not None:
        x = transforms(x)
    out = model(x, k=num_models).detach().cpu()
    out = F.log_softmax(out, dim=2)
    N, K, C = out.shape
    log_probs_N_K_C = out.to(torch.double)
    entropies = compute_entropy(log_probs_N_K_C)
    return entropies # [N]

def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)
    log_probs_N_K_C = log_probs_N_K_C.to(torch.double)
    nats_N_K_C = log_probs_N_K_C * torch.exp(log_probs_N_K_C)
    nats_N_K_C[torch.isnan(nats_N_K_C)] = 0.0
    entropies_N = -torch.sum(nats_N_K_C, dim=(1, 2)) / K
    return entropies_N

def mi_drop_out(model, x, transforms=None, num_classes=2, num_models=100):
    #https://github.com/BlackHC/active_learning_redux/blob/master/batchbald_redux/joint_entropy.py#L15
    if transforms is not None:
        x = transforms(x)
    out = model(x, k=num_models).detach().cpu()
    out = F.log_softmax(out, dim=2)
    N, K, C = out.shape
    log_probs_N_K_C = out.to(torch.double)
    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)
    return scores_N

def cross_entropy(model, x, y):
    out = model(x)
    out = out.squeeze(1)
    loss_fn = nn.CrossEntropyLoss(reduce=None)
    loss = loss_fn(out, y)
    return loss.detach()

def calc_ent_batched(model, dataloader, num_models=100):
    total_ent = 0
    total_xent = 0
    vars_ = 0
    use_cuda = True
    num_points = 0
    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()
    model.model.eval()
    # make pretrained weights to eval to avoid batchnorm batch 1 problem
    model.to(device)

    for batch_idx, out_dict in enumerate(dataloader):
        data = out_dict['data']
        target = out_dict['target']
        source_id = out_dict['source_id']
        data, target = data.to(device), target.to(device)
        #data = data.reshape(-1, 3*28*28)
        ents = entropy_drop_out(model, data, num_models=num_models)
        vars_ += torch.var(ents)
        total_ent += sum(ents)
        total_xent += torch.sum(cross_entropy(model, data, target))
        num_points += len(target)
    return (total_ent/num_points).cpu().item(), (total_xent/num_points).cpu().item(), vars_.cpu().item()

def calc_ent_per_point_batched(model, dataloader, num_models=100, mean=False, mi=False, sampled_batches=None):
    use_cuda = True
    num_points = 0
    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()
    model.model.eval()
    model.to(device)
    ents = []
    batches = 0
    for batch_idx, out_dict in enumerate(dataloader):
        batches += 1
        data = out_dict['data']

        data = data.to(device).float()
        #data = data.reshape(-1, 3*28*28)
        if mi:
            out = mi_drop_out(model, data, num_models=num_models)
        else:
            out = entropy_drop_out(model, data, num_models=num_models)
        ents.extend(out.cpu().tolist())
        if (sampled_batches != None):
            if (batches > sampled_batches):
                break

    if mean:
        return sum(ents)/len(ents)
    else:
        return ents


def calc_ent_per_source_batched(model, dataloader, num_sources, num_models=100, mi=False, return_averaged_only=True):
    total_ent = 0
    total_xent = 0
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()
    model.model.eval()
    model.to(device)
    ents = []
    sources = []
    source_ents = collections.defaultdict(float)
    for batch_idx, out_dict in enumerate(dataloader):
        data = out_dict['data']
        target = out_dict['target']
        source_id = out_dict['source_id']
        data, target = data.to(device), target.to(device)
        #data = data.reshape(-1, 3*28*28)
        if mi:
            ent = mi_drop_out(model, data, num_models=num_models)
        else:
            ent = entropy_drop_out(model, data, num_models=num_models)
        ents.extend(ent.cpu().tolist())
        sources.extend(source_id.cpu().tolist())
    for i in range(num_sources):
        source_ent = np.array(ents) @ (np.array(sources)==i)
        source_ents[i] = source_ent / (sum(np.array(sources)==i) + 1e-3)
    if return_averaged_only:
        return dict(sorted(source_ents.items()))
    else:
        return dict(sorted(source_ents.items())), ents, sources
    
def test_batched_per_source(model, dataloader_test, num_sources):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    correct_array = []
    source_array = []
    source_acc = collections.defaultdict(float)
    model.eval()
    for batch_idx, out_dict in enumerate(dataloader_test):
        data = out_dict['data']
        target = out_dict['target']
        source_id = out_dict['source_id']
        data, target = data.to(device), target.to(device)
        output = model(data).squeeze(1)
        out = output.argmax(axis=1)
        correct_array.extend((out == target).cpu().tolist())
        source_array.extend(source_id.tolist())
    for i in range(num_sources):
        source_accs = np.array(correct_array) @ (np.array(source_array)==i)
        source_acc[i] = source_accs / (sum(np.array(source_array)==i) + 1e-3)
    return source_acc

def log_dict(start_log, to_append):
    for key, value in to_append.items():
        start_log[key].append(value)
    return start_log

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

