import wandb
import json
from collections import defaultdict
import os
import scipy
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_runs(name):
    api = wandb.Api(timeout=19)
    #return api.runs('mizunt/uq_test_celeba_minprops')
    return api.runs(name)

def score_ordering_sim(dataframe_test, dataframe_ent):
    # Given two dataframes each containing rows of datasizes
    # and their corresponding test and ent values
    # return for each data size the agreement in ordering 
    # of each group in terms of test acc and entropy.
    # entropy should be inversely correlated to test acc
    order_test = {}
    order_ent = {}
    test_similarity = {}
    for row in dataframe_test.iterrows():
        sorted_column_names = tuple(row[1].sort_values(ascending=False).index.tolist())
        order_test[row[0]]= sorted_column_names
        diff = find_min_pairwise_difference(row[1].tolist())
        test_similarity[row[0]] = diff
    for row in dataframe_ent.iterrows():
        sorted_column_names = tuple(row[1].sort_values(ascending=True).index.tolist())
        order_ent[row[0]]= sorted_column_names
    dataframe_test['order'] = dataframe_test.index.map(order_test)
    dataframe_ent['order'] = dataframe_test.index.map(order_ent)
    results = {}
    for (datasize1, row1), (datasize2, row2) in zip(dataframe_test.iterrows(), dataframe_ent.iterrows()):
        assert datasize1 == datasize2
        test_order = row1['order']
        ent_order = row2['order']
        score = scipy.stats.kendalltau(test_order, ent_order)
        results[datasize1] = score.correlation
    return results, test_similarity

def find_min_pairwise_difference(arr):
    if len(arr) < 2:
        return None  # Cannot find a pair difference with less than two elements
    arr.sort()  # Step 1: Sort the list
    min_diff = float('inf')  # Step 2: Initialize min_diff to infinity

    # Step 3 & 4: Iterate and compare adjacent elements

    for i in range(1, len(arr)):
        diff = arr[i] - arr[i-1]
        if diff < min_diff:
            min_diff = diff
    
    return min_diff  # Step 5: Return minimum difference


def min_groups_above_maj(dataframe_test, dataframe_ent, maj_groups=(0,1), min_groups=(2,3)):
    # Given two dataframes each containing rows of datasizes
    # and their corresponding test and ent values
    # return for each data size the agreement in ordering 
    # of each group in terms of test acc and entropy.
    # entropy should be inversely correlated to test acc
    order_test = {}
    order_ent = {}
    for row in dataframe_test.iterrows():
        sorted_column_names = tuple(row[1].sort_values(ascending=True).index.tolist())
        order_test[row[0]]= sorted_column_names
    
    for row in dataframe_ent.iterrows():
        sorted_column_names = tuple(row[1].sort_values(ascending=True).index.tolist())
        order_ent[row[0]]= sorted_column_names
    dataframe_test['order'] = dataframe_test.index.map(order_test)
    dataframe_ent['order'] = dataframe_test.index.map(order_ent)
    results = {}
    for (datasize1, row1), (datasize2, row2) in zip(dataframe_test.iterrows(), dataframe_ent.iterrows()):
        assert datasize1 == datasize2
        test_order = row1['order']
        ent_order = row2['order']
        if maj_groups[0] in test_order[2:] and maj_groups[1] in test_order[2:]:
            # if majority groups have the highest accuracy
            if min_groups[0] in ent_order[2:] and min_groups[1] in ent_order[2:]:
                # if minority groups have the highest entropy
                score = 1
            else:
                score = 0 
        else:
            score = -1
            # not counted in result
        results[datasize1] = score
    return results

def process_results(runs, remove_column='wb ent', column_rename_ent={'mnb_ent':0,'fb_ent': 1,'fnb_ent':2,'mb_ent':3},
                    column_rename_test={'mnb test acc':0,'fb test acc': 1,'fnb test acc':2,'mb test acc':3},
                maj_groups=(0,1), min_groups=(2,3)):
    full_order = True
    results_all = {}
    for run_ in runs:
        results = run_.history()
        try:
            results = results.set_index('data size')
        except:
            continue
        entropy_keys = [item for item in results.keys() if 'ent' in item]
        entropy_keys.remove(remove_column)
        test_keys = [item for item in results.keys() if 'test' in item]
        test_vals = results[test_keys]
        ent_vals = results[entropy_keys]
        ent = ent_vals.rename(columns=column_rename_ent)
        test = test_vals.rename(
            columns=column_rename_test)
        min_prop = json.loads(run_.json_config)['minority_prop']['value']
        data_mode = json.loads(run_.json_config)['data_mode']['value']
        results_ordering, test_similarity = score_ordering_sim(test, ent)
        # for one set of results, we have agreement in ordering. 
        results['ordering_agreement'] = results.index.map(results_ordering)
        results['test min distance'] = results.index.map(test_similarity)
        results_all.update({min_prop: results})
    results_all = {key: results_all[key] for key in sorted(results_all.keys())}
    return results_all

def plot(axs, data_in, data2_in, threshold=0.01):
    ax_list = list(axs.flat)[::-1]
    for (key1, data1), (key2, data2) in zip(data_in.items(), data2_in.items()):
        try:
            ax = ax_list.pop()
        except:
            continue
        assert key1 == key2
        data_above_min_distance = data1[data1['test min distance'] > threshold]
        data_below_min_distance = data1[data1['test min distance'] < threshold]
        ax.plot(data_above_min_distance.index.to_list(), data_above_min_distance['ordering_agreement'].to_list(), 'bo')
        data_above_min_distance = data2[data2['test min distance'] > threshold]
        data_below_min_distance = data2[data2['test min distance'] < threshold]
        ax.plot(data_above_min_distance.index.to_list(), data_above_min_distance['ordering_agreement'].to_list(), 'bo')

        #ax.plot(data.index.to_list(), data['wga'].to_list(), label=min_prop)
        #ax.plot(data_below_min_distance.index.to_list(), data_below_min_distance['ordering_agreement'].to_list(), 'ro', label=min_prop)
        ax.set_title(key2)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(100, 2000)

wb = True
if wb:
    title = 'waterbirds'
    column_rename_ent = {'ww_ent':0,'ll_ent': 1,'lw_ent':2,'wl_ent':3}
    column_rename_test = {'ww test acc':0,'ll test acc': 1,'lw test acc':2,'wl test acc':3}
    remove_column = 'celeba ent'
    dino_smaller_runs = 'uq_wb_min_props_dino_smaller_size2'
    dino_larger_runs = 'uq_wb_min_props_dino'
    resnet_runs = 'uq_test_wb_uq_fixed2'

else:
    title = 'celeba'
    column_rename_ent={'mnb_ent':0,'fb_ent': 1,'fnb_ent':2,'mb_ent':3}
    column_rename_test = {'ww test acc':0,'ll test acc': 1,'lw test acc':2,'wl test acc':3}
    remove_column='wb ent'
    dino_smaller_runs = 'mizunt/min_prop_celeba_dino_smaller_size2'
    dino_larger_runs = 'mizunt/min_prop_celeba_dino'
    resnet_runs = 'uq_test_celeba_minprops'


runs = get_runs(dino_larger_runs)
runs2 = get_runs(dino_smaller_runs)
run1_results = process_results(runs, remove_column=remove_column, column_rename_ent=column_rename_ent,
                               column_rename_test=column_rename_test)
run2_results = process_results(runs2, remove_column=remove_column, column_rename_ent=column_rename_ent,
                               column_rename_test=column_rename_test)

num_plots = 8

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(6, num_plots * 2))
fig.subplots_adjust(bottom=0.1)
fig.suptitle(title, fontsize=16, fontweight='bold')

plot(axs, run1_results, run2_results)
plt.show()


