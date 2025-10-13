import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from wilds_dataset import WILDSDataset
from gdro_loss import LossComputer
import collections
from early_stopping import EarlyStopping

def train_batched(model=None, num_epochs=30, dataloader=None, dataloader_test=None,
                  weight_decay=0, lr=0.001, flatten=False, label_wb='', gdro=False, num_groups=None,
                  norm_dict=None, model_checkpoint_path='/network/scratch/m/mizu.nishikawa-toomey/waterbird_cp/',
                  wandb=False, group_mapping_fn=None, group_string_map={}, group_key='metadata'):
    now = datetime.now() 
    formatted_full = now.strftime("%A, %B %d, %Y %H:%M:%S")
    path = model_checkpoint_path + formatted_full + 'model.pt'
    early_stopping = EarlyStopping(patience=10, verbose=True, path=path)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    model.to(device)
    groups = []
    group_count_dict = collections.defaultdict(int)
    for epoch in range(num_epochs):
        print(epoch)
        total_correct = 0
        total_points = 0
        for batch_idx, data_dict in enumerate(dataloader):
            data = data_dict['data']
            target = data_dict['target']
            pseudo_g = data_dict['source_id']
            true_group = data_dict[group_key]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if flatten:
                data = data.reshape(-1, 3*28*28)
            output = model(data).squeeze(1)
            out = output.argmax(axis=1)
            if gdro:
                loss_computer = LossComputer(
                    nn.CrossEntropyLoss(reduce=False),
                    is_robust=True,
                    dataset=dataloader.dataset,
                    n_groups=num_groups,
                    alpha=0.2,
                    gamma=0.1,
                    adj=None,
                    step_size=0.01,
                    normalize_loss=False,
                    btl=False,
                    min_var_weight=0)

                loss = loss_computer.loss(output, target, pseudo_g, True)
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_correct += sum(out == target)
            total_points += len(target)
            if epoch == 0:
                groups.extend(pseudo_g)

        train_acc = (total_correct / total_points).cpu().item()
        test_acc_dict = test_per_group(model, dataloader_test,
                                       group_mapping_fn, group_string_map)
        wga = min([value for key, value in test_acc_dict.items()])
        if train_acc >0.80:
            early_stopping(-wga, model, test_acc_dict)
            if early_stopping.early_stop:
                print('early stopping at epoch ' + str(epoch))
                test_acc = early_stopping.test_acc
                wga = -early_stopping.val_loss_min
                break
            
        print('epoch ' + str(epoch))    
        print('train acc '+ str(train_acc))
        print(test_acc_dict)
        if wandb != False:
            test_acc_dict.update({'train acc': train_acc})
            test_acc_dict.update({'wga': wga})
            wandb.log(test_acc)
    for i in range(num_groups):
        # counting the number of points in each group
        # in the training data
        group_count_dict[i] = sum(np.array(groups) == i)
        
    return train_acc, test_acc_dict, group_count_dict, wga

def test_batched(model, dataloader_test, device):
    total_correct_test = 0
    total_points_test = 0
    model.eval()
    for batch_idx, data_dict in enumerate(dataloader_test):
        data = data_dict['data']
        target = data_dict['target']
        data, target = data.to(device), target.to(device)
        output = model(data).squeeze(1)
        out = output.argmax(axis=1)
        total_correct_test += sum(out == target)
        total_points_test += len(target)
    return (total_correct_test/total_points_test).cpu().item()

def test_per_group(model, test_loader, group_mapping_fn, group_string_map):
    # group mapping maps metadata to an integer that identifies a group
    # group_string_map is a dictionary mapping group integers to strings
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    correct_dict = dict.fromkeys(group_string_map.keys(), 0)
    sum_dict = dict.fromkeys(group_string_map.keys(), 1e-3)
    for data_dict in test_loader:
        data = data_dict['data']
        y = data_dict['target']
        meta = data_dict['metadata']
        group_ids = group_mapping_fn(meta)
        data, y = data.to(device), y.to(device)
        output = model(data).squeeze(1)
        correct_batch = torch.argmax(output, axis=1) == y
        for group_name, group_id in group_string_map.items():
            correct = sum(correct_batch & (group_ids.to(device) == group_id)).item()
            total = sum(group_ids == group_id).item()
            correct_dict[group_name] += correct
            sum_dict[group_name] +=total
    test_acc_final = {key + ' test acc' : correct_dict[key] / sum_dict[key] for key in correct_dict}
    return test_acc_final
