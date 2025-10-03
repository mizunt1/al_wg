import os
import pandas as pd


if __name__ == "__main__":
    root_dir = '/network/scratch/m/mizu.nishikawa-toomey/celeba'
    splits = pd.read_csv(root_dir + "/list_eval_partition.txt", sep=r"\s+", names=['image', 'split']).set_index('image')
    
    df = pd.read_csv(root_dir + "/list_attr_celeba.txt", sep=r"\s+", skiprows=1, header=0)
    merged = pd.merge(df, splits, left_index=True, right_index=True)
    train = merged[merged['split'] == 0]
    blond = train[train['Blond_Hair'] == 1]
    not_blond = train[(train['Black_Hair'] == 1) | (train['Brown_Hair'] == 1)]

    # identify group locations in train
    female_blond = blond[blond['Male'] == -1]
    male_blond = blond[blond['Male'] == 1]

    male_notblond = not_blond[not_blond['Male'] == 1]
    female_notblond = not_blond[not_blond['Male'] == -1]
    merged['groups'] = merged['split']
    merged.loc[male_blond.index, 'groups'] = 3
    merged.loc[female_blond.index, 'groups'] = 4
    merged.loc[male_notblond.index, 'groups'] = 5
    merged.loc[female_notblond.index, 'groups'] = 6
    print('train data')
    print(f"num male blond: {len(merged[merged['groups'] == 3])}")
    print(f'num female blond: {len(merged[merged["groups"] == 4])}')
    print(f'num male not blond : {len(merged[merged["groups"] == 5])}')
    print(f'num female not blond: {len(merged[merged["groups"] == 6])}')

    # num male blond: 1260
    # num female blond: 22020
    # num male not blond : 30103
    # num female not blond: 41064
    test = merged[merged['split'] == 2]
    blond = test[test['Blond_Hair'] == 1]
    not_blond = test[(test['Black_Hair'] == 1) | (train['Brown_Hair'] == 1)]

    # identify group locations in train
    female_blond = blond[blond['Male'] == -1]
    male_blond = blond[blond['Male'] == 1]

    male_notblond = not_blond[not_blond['Male'] == 1]
    female_notblond = not_blond[not_blond['Male'] == -1]
    merged['groups'] = merged['split']
    merged.loc[male_blond.index, 'groups'] = 7
    merged.loc[female_blond.index, 'groups'] = 8
    merged.loc[male_notblond.index, 'groups'] = 9
    merged.loc[female_notblond.index, 'groups'] = 10
    print('test data')
    print(f"num male blond: {len(merged[merged['groups'] == 7])}")
    print(f'num female blond: {len(merged[merged["groups"] == 8])}')
    print(f'num male not blond : {len(merged[merged["groups"] == 9])}')
    print(f'num female not blond: {len(merged[merged["groups"] == 10])}')


    merged['groups'].to_csv(root_dir + '/celeb_a_labelled_groups.txt', sep=' ', index=True, header=False)
    merged_read = pd.read_csv(root_dir + '/celeb_a_labelled_groups.csv')

    
