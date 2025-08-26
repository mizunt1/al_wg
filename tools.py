import os
import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

