
import os
import numpy as np
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dataset_utils import crop_and_resize, combine_and_mask

np.random.seed(0)
random.seed(0)
################ Paths and other configs - Set these #################################
cub_dir = '/network/scratch/m/mizu.nishikawa-toomey/CUB_200_2011/'
# tar -xf /network/datasets/places365/large/train_large_places365standard.tar -C /tmp/
places_dir = '/tmp/'
places_dir_info = '/network/datasets/places365/'
output_dir = '/network/scratch/m/mizu.nishikawa-toomey/waterbird_larger/'

dataset_name = 'waterbird_larger'

target_places_list = [
    ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
    ['ocean', 'lake/natural']]              # Water backgrounds

val_frac = 0.2             # What fraction of the training data to use as validation
confounder_strength = 0.5 # Determines relative size of majority vs. minority groups
######################################################################################

images_path = os.path.join(cub_dir, 'images.txt')

df = pd.read_csv(
    images_path,
    sep=" ",
    header=None,
    names=['img_id', 'img_filename'],
    index_col='img_id')

### Set up labels of waterbirds vs. landbirds
# We consider water birds = seabirds and waterfowl.
species = np.unique([img_filename.split('/')[0].split('.')[1].lower() for img_filename in df['img_filename']])
water_birds_list = [
    'Albatross', # Seabirds
    'Auklet',
    'Cormorant',
    'Frigatebird',
    'Fulmar',
    'Gull',
    'Jaeger',
    'Kittiwake',
    'Pelican',
    'Puffin',
    'Tern',
    'Gadwall', # Waterfowl
    'Grebe',
    'Mallard',
    'Merganser',
    'Guillemot',
    'Pacific_Loon'
]

water_birds = {}
for species_name in species:
    water_birds[species_name] = 0
    for water_bird in water_birds_list:
        if water_bird.lower() in species_name:
            water_birds[species_name] = 1
species_list = [img_filename.split('/')[0].split('.')[1].lower() for img_filename in df['img_filename']]
df['y'] = [water_birds[species] for species in species_list]
### Assign train/tesst/valid splits
# In the original CUB dataset split, split = 0 is test and split = 1 is train
# We want to change it to
# split = 0 is train,
# split = 1 is val,
# split = 2 is test
waterbirds = df[df['y'] == 1]
landbirds = df[df['y'] == 0]
df['split'] = 0
num_val = 400
num_test = 400
# give 400 from each label for validation and test
waterbirds_val_test_idx = np.sort(np.random.choice(waterbirds.index, size=num_val, replace=False))
landbirds_val_test_idx = np.sort(np.random.choice(landbirds.index, size=num_test, replace=False))

wb_val_idx = waterbirds_val_test_idx[:num_val//2]
wb_test_idx = waterbirds_val_test_idx[num_val//2:]

lb_val_idx = landbirds_val_test_idx[:num_test//2]
lb_test_idx = landbirds_val_test_idx[num_test//2:]

val_idx = np.concatenate((wb_val_idx, lb_val_idx))
test_idx = np.concatenate((wb_test_idx, lb_test_idx))

df.loc[val_idx, 'split'] = 1
df.loc[test_idx, 'split'] = 2
### Assign confounders (place categories)

# Confounders are set up as the following:
# Y = 0, C = 0: confounder_strength
# Y = 0, C = 1: 1 - confounder_strength
# Y = 1, C = 0: 1 - confounder_strength
# Y = 1, C = 1: confounder_strength

df['place'] = 0
train_ids = np.array(df.loc[df['split'] == 0].index)
val_ids = np.array(df.loc[df['split'] == 1].index)
test_ids = np.array(df.loc[df['split'] == 2].index)
for split_idx, ids in enumerate([train_ids, val_ids, test_ids]):
    for y in (0,1):
        # In original code for train and val, data is confounded
        # but test is not. In this code we have changed it so
        # in val it is also unconfounded. 
        if split_idx == 0: # train
            if y == 0:
                pos_fraction = 1 - confounder_strength
            else:
                pos_fraction = confounder_strength
        else:
            pos_fraction = 0.5
        subset_df = df.loc[ids, :]
        y_ids = np.array((subset_df.loc[subset_df['y'] == y]).index)
        pos_place_ids = np.random.choice(
            y_ids,
            size=int(np.round(pos_fraction * len(y_ids))),
            replace=False)
        df.loc[pos_place_ids, 'place'] = 1

for split, split_label in [(0, 'train'), (1, 'val'), (2, 'test')]:
    print(f"{split_label}:")
    split_df = df.loc[df['split'] == split, :]
    print(f"waterbirds are {np.mean(split_df['y']):.3f} of the examples")
    print(f"y = 0, c = 0: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 0))}")
    print(f"y = 0, c = 1: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 1))}")
    print(f"y = 1, c = 0: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 0))}")
    print(f"y = 1, c = 1: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 1))}")


### Assign places to train, val, and test set
place_ids_df = pd.read_csv(
    os.path.join(places_dir_info, 'categories_places365.txt'),
    sep=" ",
    header=None,
    names=['place_name', 'place_id'],
    index_col='place_id')

target_place_ids = []

for idx, target_places in enumerate(target_places_list):
    place_filenames = []

    for target_place in target_places:
        target_place_full = f'/{target_place[0]}/{target_place}'
        assert (np.sum(place_ids_df['place_name'] == target_place_full) == 1)
        target_place_ids.append(place_ids_df.index[place_ids_df['place_name'] == target_place_full][0])
        print(f'train category {idx} {target_place_full} has id {target_place_ids[idx]}')

        # Read place filenames associated with target_place
        place_filenames += [
            f'/{target_place[0]}/{target_place}/{filename}' for filename in os.listdir(
                os.path.join(places_dir, 'data_large', target_place[0], target_place))
            if filename.endswith('.jpg')]

    random.shuffle(place_filenames)

    # Assign each filename to an image
    indices = (df.loc[:, 'place'] == idx)
    assert len(place_filenames) >= np.sum(indices),\
        f"Not enough places ({len(place_filenames)}) to fit the dataset ({np.sum(df.loc[:, 'place'] == idx)})"
    df.loc[indices, 'place_filename'] = place_filenames[:np.sum(indices)]

### Write dataset to disk
output_subfolder = os.path.join(output_dir, dataset_name)
os.makedirs(output_subfolder, exist_ok=True)

df.to_csv(os.path.join(output_subfolder, 'metadata.csv'))

for i in tqdm(df.index):
    # Load bird image and segmentation
    img_path = os.path.join(cub_dir, 'images', df.loc[i, 'img_filename'])
    seg_path = os.path.join(cub_dir, 'segmentations/bin_mask', df.loc[i, 'img_filename'].replace('.jpg','.png'))
    img_np = np.asarray(Image.open(img_path).convert('RGB'))
    seg_np = np.asarray(Image.open(seg_path)) 
    seg_np = np.repeat(seg_np[:,:,np.newaxis], 3, axis=2) 
    # Load place background
    # Skip front /
    place_path = os.path.join(places_dir, 'data_large', df.loc[i, 'place_filename'][1:])
    place = Image.open(place_path).convert('RGB')
    img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
    combined_img = combine_and_mask(place, seg_np, img_black)

    output_path = os.path.join(output_subfolder, df.loc[i, 'img_filename'])
    os.makedirs('/'.join(output_path.split('/')[:-1]), exist_ok=True)

    combined_img.save(output_path)
