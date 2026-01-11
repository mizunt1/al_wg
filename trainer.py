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
import time
def train_batched(model=None, num_epochs=30, dataloader=None, dataloader_test=None, dataloader_val=None,
                  weight_decay=0, lr=0.001, flatten=False, gdro=False, num_groups=None,
                  num_sources=None,
                  model_checkpoint_path='/network/scratch/m/mizu.nishikawa-toomey/checkpoints/',
                  wandb=False, group_mapping_fn=None, group_string_map=None, group_string_map_test=None,
                  group_key='metadata',
                  true_group_in_loss=False, sample_batch_val=None):
    if group_string_map_test == None:
        group_string_map_test = group_string_map
    train_acc_has_surpassed = False
    now = datetime.now()
    early_stopping = EarlyStopping(patience=20, verbose=True, path=model_checkpoint_path)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    model.to(device)
    groups = []
    sources = []
    group_count_dict = collections.defaultdict(int)
    source_count_dict = collections.defaultdict(int)
    for epoch in range(num_epochs):        
        print(epoch)
        total_correct = 0
        total_points = 0
        for batch_idx, data_dict in enumerate(dataloader):
            data = data_dict['data']
            target = data_dict['target']
            source_id = data_dict['source_id']
            true_group = data_dict[group_key]
            group = group_mapping_fn(true_group)
            data, target = data.to(device), target.to(device)
            if true_group_in_loss:
                group_in_loss = group
            else:
                group_in_loss = source_id
                    
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
                    n_groups=num_sources,
                    alpha=0.2,
                    gamma=0.1,
                    adj=None,
                    step_size=0.01,
                    normalize_loss=False,
                    btl=False,
                    min_var_weight=0)
                loss = loss_computer.loss(output, target, group_in_loss, True)

            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_correct += sum(out == target).cpu()
            total_points += len(target)
            if epoch == 0:
                groups.extend(group)
                sources.extend(source_id)
        train_acc = (total_correct / total_points).item()
        if train_acc >0.80:
            train_acc_has_surpassed = True
            # only calculate test acc on epochs when acc is high to save on computing it
            val_acc_dict = test_per_group(model, dataloader_val,
                                          group_mapping_fn, group_string_map_test,
                                          sampled_batches=sample_batch_val,
                                          string_append=' val')
            wga = min([value for key, value in val_acc_dict.items()])
            early_stopping(-wga, model, val_acc_dict)
            if early_stopping.early_stop:
                print('early stopping at epoch ' + str(epoch))
                val_acc_dict = early_stopping.val_acc_dict
                wga = -early_stopping.val_loss_min
                break
            print('val acc dict: ')
            print(val_acc_dict)

        print('epoch ' + str(epoch))    
        print('train acc '+ str(train_acc))
        print('loss' + str(loss.detach()))

        if wandb != False:
            # for logging each epoch for training
            val_acc_dict.update({'train acc': train_acc})
            val_acc_dict.update({'wga': wga})
            wandb.log(wga)
    for i in range(num_groups):
        # counting the number of points in each group
        # in the training data
        group_count_dict[i] = sum(np.array(groups) == i)
    for i in range(num_sources):
        # counting the number of points in each group
        # in the training data
        source_count_dict[i] = sum(np.array(sources) == i)
    if not train_acc_has_surpassed:
        # if train acc has not surpassed 80 through all epochs, need to calc test acc here
        val_acc_dict = test_per_group(model, dataloader_val,
                                      group_mapping_fn, group_string_map_test,
                                      sampled_batches=sample_batch_val,
                                      string_append=' val')
        wga = min([value for key, value in val_acc_dict.items()])
        early_stopping.save_checkpoint(wga, model, wga)
        print('val acc dict final')
        print(val_acc_dict)
    # final test accuracy calculated on checkpointed model
    model.load_state_dict(torch.load(early_stopping.path, weights_only=True))
    model.eval()
    test_acc_dict = test_per_group(model, dataloader_test,
                                   group_mapping_fn, group_string_map_test,
                                   string_append=' test')
    wga_test = min([value for key, value in test_acc_dict.items()])
    print('test acc dict: ')
    print(test_acc_dict)
    
    return train_acc, val_acc_dict, test_acc_dict, group_count_dict, source_count_dict, wga, wga_test

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

def test_per_group(model, test_loader, group_mapping_fn, group_string_map,
                   sampled_batches=None, string_append=' '):
    # group mapping maps metadata to an integer that identifies a group
    # group_string_map is a dictionary mapping group integers to strings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    correct_dict = dict.fromkeys(group_string_map.keys(), 0)
    sum_dict = dict.fromkeys(group_string_map.keys(), 1e-3)
    batches = 0
    y_ = []
    colour = []
    groups_ = []
    for data_dict in test_loader:
        batches += 1
        data = data_dict['data']
        y = data_dict['target']
        meta = data_dict['metadata']
        y_.append(meta[0])
        colour.append(meta[1])
        group_ids = group_mapping_fn(meta)
        groups_.append(group_ids)
        data, y = data.to(device), y.to(device)
        output = model(data).squeeze(1)
        correct_batch = torch.argmax(output, axis=1) == y
        for group_name, group_id in group_string_map.items():
            correct = sum(correct_batch & (group_ids.to(device) == group_id)).item()
            total = sum(group_ids == group_id).item()
            correct_dict[group_name] += correct
            sum_dict[group_name] +=total
        if (sampled_batches != None):
            if (batches > sampled_batches):
                break
    test_acc_final = {key +string_append : correct_dict[key] / sum_dict[key] for key in correct_dict}
    #import pdb
    #pdb.set_trace()
    print('sum dict:')
    print(sum_dict)
    return test_acc_final


def trainer_erm(model, dataloader, dataloader_test=None, num_epochs=10, lr=1e-3,
                weight_decay=0, device=None, checkpoint_path=None, verbose=True):
    """Simple ERM trainer: standard cross-entropy training loop with early stopping.

    Returns (model, train_acc, test_acc)
    - model: trained model (moved to device)
    - train_acc: final training accuracy (float)
    - test_acc: test accuracy if dataloader_test provided else None
    """
    use_cuda = torch.cuda.is_available()
    device = device or (torch.device('cuda') if use_cuda else torch.device('cpu'))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    train_acc_has_surpassed = False
    early_stopping = None
    test_acc = None
    
    if dataloader_test is not None and checkpoint_path is not None:
        early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint_path)

    for epoch in range(num_epochs):
        model.train()
        total_correct = 0
        total_points = 0
        for batch_idx, data_dict in enumerate(dataloader):
            data = data_dict['data']
            target = data_dict['target']
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data).squeeze(1)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_correct += (output.argmax(dim=1) == target).sum().item()
            total_points += len(target)
        train_acc = (total_correct / total_points) if total_points > 0 else 0.0
        
        if train_acc > 0.8:
            train_acc_has_surpassed = True
            if early_stopping is not None:
                test_acc = test_batched(model, dataloader_test, device)
                early_stopping(-test_acc, model, test_acc_dict={})
                if early_stopping.early_stop:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    test_acc = -early_stopping.val_loss_min
                    break
                if verbose:
                    print(f"Epoch {epoch}: train acc {train_acc:.4f}, test acc {test_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch}: train acc {train_acc:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch}: train acc {train_acc:.4f}")

    if not train_acc_has_surpassed and dataloader_test is not None:
        test_acc = test_batched(model, dataloader_test, device)
        if verbose:
            print(f"Test acc: {test_acc:.4f}")

    return model, train_acc, test_acc
