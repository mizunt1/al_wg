import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
import numpy as np
from wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from gdro_loss import LossComputer
import collections
from early_stopping import EarlyStopping
class WaterbirdsDataset(WILDSDataset):
    """
    The Waterbirds dataset.
    This dataset is not part of the official WILDS benchmark.
    We provide it for convenience and to facilitate comparisons to previous work.

    Supported `split_scheme`:
        'official'

    Input (x):
        Images of birds against various backgrounds that have already been cropped and centered.

    Label (y):
        y is binary. It is 1 if the bird is a waterbird (e.g., duck), and 0 if it is a landbird.

    Metadata:
        Each image is annotated with whether the background is a land or water background.

    Original publication:
        @inproceedings{sagawa2019distributionally,
          title = {Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle = {International Conference on Learning Representations},
          year = {2019}
        }

    The dataset was constructed from the CUB-200-2011 dataset and the Places dataset:
        @techreport{WahCUB_200_2011,
        	Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
        	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
        	Year = {2011}
        	Institution = {California Institute of Technology},
        	Number = {CNS-TR-2011-001}
        }
        @article{zhou2017places,
          title = {Places: A 10 million Image Database for Scene Recognition},
          author = {Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
          journal ={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          year = {2017},
          publisher = {IEEE}
        }

    License:
        The use of this dataset is restricted to non-commercial research and educational purposes.
    """

    _dataset_name = 'waterbirds'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/',
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official',
                 metadata_name='metadata.csv', rep_file_path='data/waterbirds_v1.0/waterbirds_resnet50',
                 split_names={'train': 'Train', 'val': 'Validation', 'test': 'Test'}, use_rep=False):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        # Note: metadata_df is one-indexed.
        metadata_df = pd.read_csv(
            os.path.join(self.data_dir, metadata_name))
        self.rep_file_path = rep_file_path
        # Get the y values
        self._y_array = torch.LongTensor(metadata_df['y'].values)
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata_df['place'].values), self._y_array),
            dim=1
        )
        self._metadata_fields = ['background', 'y']
        self._metadata_map = {
            'background': [' land', 'water'], # Padding for str formatting
            'y': [' landbird', 'waterbird']
        }

        # Extract filenames
        self._input_array = metadata_df['img_filename'].values
        self._original_resolution = (224, 224)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            self.DEFAULT_SPLITS = split_scheme
            self.DEFAULT_SPLIT_NAMES = split_names
        self._split_dict = split_scheme
        self._split_names = split_names
        self._split_array = metadata_df['split'].values
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['background', 'y']))

        self.use_rep = use_rep
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
       """
       Returns x for a given idx.
       """
       if self.use_rep:
           file_name = self._input_array[idx].split('jpg')[0].replace('/','')
           x = torch.load(f"{self.rep_file_path}/tensor_{file_name}pt", weights_only=False)
       else:
           img_filename = os.path.join(
               self.data_dir,
               self._input_array[idx])
           x = Image.open(img_filename).convert('RGB')
       return x

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)

        results, results_str = self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

        # For Waterbirds, the validation and test sets are constructed to be more balanced
        # compared to the training set.
        # To compute the actual average accuracy over the empirical (training) distribution,
        # we therefore weight each groups according to their frequency in the training set.

        results['adj_acc_avg'] = (
            (results['acc_y:landbird_background:land'] * 3498
            + results['acc_y:landbird_background:water'] * 184
            + results['acc_y:waterbird_background:land'] * 56
            + results['acc_y:waterbird_background:water'] * 1057) /
            (3498 + 184 + 56 + 1057))

        del results['acc_avg']
        results_str = f"Adjusted average acc: {results['adj_acc_avg']:.3f}\n" + '\n'.join(results_str.split('\n')[1:])

        return results, results_str
    
def train_batched(model=None, num_epochs=30, dataloader=None, dataloader_test=None,
                  weight_decay=0, lr=0.001, flatten=False, label_wb='', gdro=False, num_groups=None,
                  norm_dict=None, model_checkpoint_path='/network/scratch/m/mizu.nishikawa-toomey/waterbird_cp/', wandb=False):
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
    group_dict = collections.defaultdict(int)
    for epoch in range(num_epochs):
        print(epoch)
        total_correct = 0
        total_points = 0
        for batch_idx, data_dict in enumerate(dataloader):
            data = data_dict['data']
            target = data_dict['target']
            meta = data_dict['metadata']
            pseudo_g = data_dict['group_id']
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
        test_acc = test_per_group(model, dataloader_test, ' ')
        wga = min([value for key, value in test_acc.items()])
        if train_acc >0.80:
            early_stopping(-wga, model, test_acc)
            if early_stopping.early_stop:
                print('early stopping at epoch ' + str(epoch))
                test_acc = early_stopping.test_acc
                wga = -early_stopping.val_loss_min
                break
            
        print('epoch ' + str(epoch))    
        print('train acc '+ str(train_acc))
        print(test_acc)
        if wandb != False:
            test_acc.update({'train acc': train_acc})
            test_acc.update({'wga': wga})
            wandb.log(test_acc)
    for i in range(num_groups):
        group_dict[i] = sum(np.array(groups) == i)
        
    return train_acc, test_acc, group_dict, wga

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

def test_per_group(model, test_loader, annot):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    ww = 0
    ww_total_sum = 1e-3
    ll = 0
    ll_total_sum = 1e-3
    wl = 0
    wl_total_sum = 1e-3
    lw = 0
    lw_total_sum = 1e-3
    correct = 0
    total_data = 0
    for data_dict in test_loader:
        data = data_dict['data']
        y = data_dict['target']
        meta = data_dict['metadata']
        pseudo_g = data_dict['group_id']
        data, y = data.to(device), y.to(device)
        output = model(data).squeeze(1)
        correct_batch = torch.argmax(output, axis=1) == y
        correct += correct_batch.sum()
        wwb, ww_total_sumb, llb, ll_total_sumb, wlb, wl_total_sumb, lwb, lw_total_sumb = return_group_acc(correct_batch, meta)
        ww += wwb
        ww_total_sum += ww_total_sumb
        ll += llb
        ll_total_sum += ll_total_sumb
        wl += wlb
        wl_total_sum += wl_total_sumb
        lw += lwb
        lw_total_sum += lw_total_sumb
    test_acc_final = {annot + ' ww test acc': (ww/ww_total_sum).cpu().item(), annot + ' ll test acc': (ll/ll_total_sum).cpu().item(), annot + ' wl test acc': (wl/wl_total_sum).cpu().item(), annot + ' lw test acc' : (lw/lw_total_sum).cpu().item()}
    #print('Test accuracy for ww: {:.3f}, ll: {:.3f}, wbl: {:.3f}, lbw: {:.3f}'.format(
    #    ww/ww_total_sum, ll/ll_total_sum, wl/wl_total_sum, lw/lw_total_sum))
    #print('Test, total count for ww: {:.1f}, ll: {:.1f}, wbl: {:.1f}, lbw: {:.1f}'.format(
    #    ww_total_sum, ll_total_sum, wl_total_sum, lw_total_sum))
    #print('Test total pred count for ww: {:.1f}, ll: {:.1f}, wbl: {:.1f}, lbw: {:.1f}'.format(
    #    ww, ll, wl, lw))
    return test_acc_final


def return_group_acc(correct_list, metadata_list):
    background = 0
    label = 1
    land = 0
    water = 1

    ww = 0
    ww_total_sum = 1e-3
    ll = 0
    ll_total_sum = 1e-3
    wl = 0
    wl_total_sum = 1e-3
    lw = 0
    lw_total_sum = 1e-3
    correct = 0
    total_data = 0

    for correct, metadata in zip(correct_list, metadata_list):
        if metadata[background] == land and metadata[label] == land:
            ll_total_sum += 1
            ll += correct*1
        elif metadata[background] == land and metadata[label] == water:
            wl_total_sum += 1
            wl += correct*1
        elif metadata[background] == water and metadata[label] == water:
            ww_total_sum += 1
            ww += correct*1
        else:
            lw_total_sum += 1
            lw += correct*1
    return ww, ww_total_sum, ll, ll_total_sum, wl, wl_total_sum, lw, lw_total_sum
# Prepare the standard data loader

def count_groups(data):
    ww = 0
    wl = 0
    ll = 0
    lw = 0
    background = 0
    label = 1
    land = 0
    water = 1
    for data in data.metadata_array:
        if data[background] == land and data[label] == land:
            ll += 1
        elif data[background] == land and data[label] == water:
            wl += 1
        elif data[background] == water and data[label] == water:
            ww += 1
        else:
            lw += 1
    counts = {'ww': ww, 'wl': wl,'ll':ll,'lw':lw}
    return counts
        

