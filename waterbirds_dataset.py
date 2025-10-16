import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
import numpy as np
from wilds_dataset import WILDSDataset

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
        if version != 'larger':
            self._data_dir = self.initialize_data_dir(root_dir, download)
        else:
            self._data_dir = root_dir
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
        self.use_rep = use_rep
        super().__init__(root_dir, download, split_scheme)
        self.group_string_map = {'ww': 0, 'll': 1, 'wl': 2, 'lw': 3}

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

    def group_mapping_fn(self, metadata):
        # given a batch of metadata, return a batch of integer group ids
        background = 0
        label = 1
        land = 0
        water = 1
        ww =  (metadata[:, background] == water) & (metadata[:, label] == water)
        ll =  (metadata[:, background] == land) & (metadata[:, label] == land)
        wl =  (metadata[:, background] == land) & (metadata[:, label] == water)
        lw =  (metadata[:, background] == water) & (metadata[:, label] == land)
        groups = torch.zeros(len(metadata))
        groups = groups + ll + wl*2 + lw*3
        return groups


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
        
def count_groups_dataloader(test_loader):
    ww_total_sum = 0
    ll_total_sum = 0
    wl_total_sum = 0
    lw_total_sum = 0
    background = 0
    label = 1
    land = 0
    water = 1    
    for data_dict in test_loader:
        data = data_dict['data']
        y = data_dict['target']
        meta = data_dict['metadata']
        pseudo_g = data_dict['group_id']
        ll = sum((meta[:,background] == land) & (meta[:,label] == land))
        ww = sum((meta[:,background] == water) & (meta[:,label] == water))
        lw = sum((meta[:,background] == water) & (meta[:,label] == land))
        wl = sum((meta[:,background] == land) & (meta[:,label] == water))
        ww_total_sum += ww
        ll_total_sum += ll
        wl_total_sum += wl
        lw_total_sum += lw
    return {'ww': ww_total_sum, 'll': ll_total_sum, 'wl': wl_total_sum, 'lw': lw_total_sum}

