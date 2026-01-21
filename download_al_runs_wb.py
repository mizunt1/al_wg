import wandb
import json
from collections import defaultdict
import os
import scipy
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def get_runs(name, filters):
    api = wandb.Api(timeout=19)
    #return api.runs('mizunt/uq_test_celeba_minprops')
    return api.runs(name, filters=filters)

def plot_wga_by_acquisition(name, acquisitions=None):
    """
    dataframes: list of pd.DataFrames
    config_files: list of paths to json configs
    """

    all_rows = []

    # ------------------------------------------------
    # 1. Load dataframes + configs into a single table
    # ------------------------------------------------    
    if 'cam' not in name:
        runs = get_runs(name, filters={"config.n_maj_sources":3})
    else:
        runs = get_runs(name, filters={})
    for run_ in runs:
        df = run_.history()
        json_cfg = json.loads(run_.json_config)
        
        acq = json_cfg.get("acquisition", None)
        if acq is None:
            continue  # skip if no acquisition type
        acq = acq['value']
        # Each df contains multiple rows with "num points" and "wga test"
        if "num points" not in df.columns or "wga test" not in df.columns:
            continue

        for _, row in df.iterrows():
            all_rows.append({
                "acquisition": acq,
                "num_points": row["num points"],
                "wga test": row["wga test"]
            })

    # Convert into one DataFrame
    full_df = pd.DataFrame(all_rows)
    # ------------------------------------------------
    # 2. Group by acquisition type and num_points
    # ------------------------------------------------
    agg = (full_df
           .groupby(["acquisition", "num_points"])
           .agg(mean_wga=("wga test", "mean"),
                std_wga=("wga test", "std"))
           .reset_index())

    # ------------------------------------------------
    # 3. Plot with distinct colors (tab20 supports up to 20 colors)
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    if acquisitions is None:
        acquisitions = sorted(agg["acquisition"].unique())
    n_acq = len(acquisitions)

    # determine common plotting cutoff: compute max num_points per acquisition, then take min of those maxima
    max_per_acq = {}
    for acq in acquisitions:
        sub_max = agg.loc[agg["acquisition"] == acq, "num_points"]
        if sub_max.empty:
            max_per_acq[acq] = -1
        else:
            max_per_acq[acq] = int(sub_max.max())
    # only consider acquisitions that have data (max >= 0)
    valid_maxes = [v for v in max_per_acq.values() if v >= 0]
    if not valid_maxes:
        print("No valid num_points ranges to plot.")
        return
    common_max = min(valid_maxes)

    # assign fixed colors for specific acquisition names
    reserved_colors = {
        "random": "black",
        "uniform_sources": "orange",
        "random_gdro": "red",
        "entropy": "hotpink",
        'entropy_per_source_soft_rank': "blue",
        'n_largest_soft_rank': "green"
    }

    # map acquisition names for legend display
    acq_name_map = {
        "n_largest_soft_rank": "EntPerSourceTopM-SoftRank",
        "entropy_per_source_soft_rank": "EntPerSourceProb-SoftRank"
    }

    # build final color map: reserved names get fixed colors; others get distinct hues
    color_map = {}
    for acq in acquisitions:
        if acq in reserved_colors:
            color_map[acq] = reserved_colors[acq]

    other_acqs = [a for a in acquisitions if a not in color_map]
    n_other = len(other_acqs)
    if n_other > 0:
        if n_other <= 20:
            cmap = mpl.cm.get_cmap("tab20")
            palette = [cmap(i) for i in range(n_other)]
        else:
            cmap1 = mpl.cm.get_cmap("tab20")
            cmap2 = mpl.cm.get_cmap("tab20b")
            cmap3 = mpl.cm.get_cmap("tab20c")
            palette = ([cmap1(i) for i in range(20)] +
                       [cmap2(i) for i in range(20)] +
                       [cmap3(i) for i in range(20)])[:n_other]
        for acq, col in zip(other_acqs, palette):
            color_map[acq] = col

    for idx, acq in enumerate(acquisitions):
        # sub = agg[agg["acquisition"] == acq].sort_values("num_points")
        # restrict to the common range so all acquisitions plot to the same x (<= common_max)
        sub = agg[(agg["acquisition"] == acq) & (agg["num_points"] <= common_max)].sort_values("num_points")
 
        x = sub["num_points"].to_numpy()
        y = sub["mean_wga"].to_numpy()
        std = sub["std_wga"].to_numpy()
 
        color = color_map.get(acq, None)
        label = acq_name_map.get(acq, acq)
        ax.plot(x, y, marker="o", label=label, color=color)
        ax.fill_between(x, y - std, y + std, alpha=0.2, color=color)

    ax.set_xlabel("Num Points")
    ax.set_ylabel("WGA Test")
    ax.set_title("WGA Test vs Num Points (Mean ± Std Across Experiments), Grouped by Acquisition Type")
    ax.legend(title="Acquisition", loc='lower right')
    ax.grid(True)
    fig.tight_layout()
    
    out_path = f"wga_by_acquisition_{name}.png"
    fig.canvas.draw()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)

def plot_test_acc_range_by_num_points(name, acquisitions=None):
    """
    Plot the range (max - min) of test accuracy columns across num_points.#
    
    For each run, find all columns ending with 'test acc', compute the range
    of values across those columns for each num_points, then aggregate and plot.
    """
    
    all_rows = []
    
    # ------------------------------------------------
    # 1. Load dataframes + configs into a single table
    # ------------------------------------------------
    if 'cam' not in name:
        runs = get_runs(name, filters={"config.n_maj_sources":3})
    else:
        runs = get_runs(name, filters={})
    for run_ in runs:
        df = run_.history()
        json_cfg = json.loads(run_.json_config)
        
        acq = json_cfg.get("acquisition", None)
        if acq is None:
            continue
        acq = acq['value']
        # Find all columns ending with 'test acc'
        test_acc_cols = [col for col in df.columns if col.endswith('test')]
        test_acc_cols = [col for col in test_acc_cols if 'wga' not in col]
        test_acc_cols = [col for col in test_acc_cols if 'ent' not in col]
        if not test_acc_cols or "num points" not in df.columns:
            continue
        
        # For each row, compute range across test acc columns
        for _, row in df.iterrows():
            num_points = row["num points"]
            test_acc_values = [row[col] for col in test_acc_cols if pd.notna(row[col])]
            
            if test_acc_values:
                acc_range = max(test_acc_values) - min(test_acc_values)
                all_rows.append({
                    "acquisition": acq,
                    "num_points": num_points,
                    "test_acc_range": acc_range
                })
    
    # Convert into one DataFrame
    full_df = pd.DataFrame(all_rows)
    if full_df.empty:
        print("No data collected (empty dataframe).")
        return
    
    # ------------------------------------------------
    # 2. Group by acquisition type and num_points, compute mean range
    # ------------------------------------------------
    agg = (full_df
           .groupby(["acquisition", "num_points"])
           .agg(mean_range=("test_acc_range", "mean"),
                std_range=("test_acc_range", "std"))
           .reset_index())
    
    # ------------------------------------------------
    # 3. Plot with distinct colors (tab20 supports up to 20 colors)
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    if acquisitions is None:
        acquisitions = sorted(agg["acquisition"].unique())
    n_acq = len(acquisitions)
    max_per_acq = {}
    for acq in acquisitions:
        sub_max = agg.loc[agg["acquisition"] == acq, "num_points"]
        if sub_max.empty:
            max_per_acq[acq] = -1
        else:
            max_per_acq[acq] = int(sub_max.max())
    # only consider acquisitions that have data (max >= 0)
    valid_maxes = [v for v in max_per_acq.values() if v >= 0]
    if not valid_maxes:
        print("No valid num_points ranges to plot.")
        return
    common_max = min(valid_maxes)

    # assign fixed colors for specific acquisition names
    reserved_colors = {
        "random": "red",
        "uniform_sources": "orange",
        "random_gdro": "black",
        "entropy": "hotpink",
        "entropy_per_source_soft_rank": "blue",
        "n_largest_soft_rank": "green",
    }

    # map acquisition names for legend display
    acq_name_map = {
        "n_largest_soft_rank": "EntPerSourceTopM-SoftRank",
        "entropy_per_source_soft_rank": "EntPerSourceProb-SoftRank"
    }

    # build final color map: reserved names get fixed colors; others get distinct hues
    color_map = {}
    for acq in acquisitions:
        if acq in reserved_colors:
            color_map[acq] = reserved_colors[acq]

    other_acqs = [a for a in acquisitions if a not in color_map]
    n_other = len(other_acqs)
    if n_other > 0:
        if n_other <= 20:
            cmap = mpl.cm.get_cmap("tab20")
            palette = [cmap(i) for i in range(n_other)]
        else:
            cmap1 = mpl.cm.get_cmap("tab20")
            cmap2 = mpl.cm.get_cmap("tab20b")
            cmap3 = mpl.cm.get_cmap("tab20c")
            palette = ([cmap1(i) for i in range(20)] +
                       [cmap2(i) for i in range(20)] +
                       [cmap3(i) for i in range(20)])[:n_other]
        for acq, col in zip(other_acqs, palette):
            color_map[acq] = col

    for idx, acq in enumerate(acquisitions):
        # sub = agg[agg["acquisition"] == acq].sort_values("num_points")
        # restrict to the common range so all acquisitions plot to the same x (<= common_max)
        sub = agg[(agg["acquisition"] == acq) & (agg["num_points"] <= common_max)].sort_values("num_points")
        
        x = sub["num_points"].to_numpy()
        y = sub["mean_range"].to_numpy()
        std = sub["std_range"].to_numpy()
        
        color = color_map.get(acq, None)
        label = acq_name_map.get(acq, acq)
        ax.plot(x, y, marker="o", label=label, color=color)
        ax.fill_between(x, y - std, y + std, alpha=0.2, color=color)
        
    ax.set_xlabel("Num Points")
    ax.set_ylabel("Range of Test Acc")
    ax.set_title("Range of Test Accuracy vs Num Points (Mean ± Std)")
    ax.legend(title="Acquisition", bbox_to_anchor=(1.05, 1), loc='upper right')  # Changed to upper right
    ax.grid(True)
    fig.tight_layout()
    
    out_path = f"test_acc_range_{name}.png"
    fig.canvas.draw()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)

def plot_wga_by_acquisition_cmnist_ablations(name, acquisitions=None):
    """
    dataframes: list of pd.DataFrames
    config_files: list of paths to json configs
    """

    all_rows = []

    # ------------------------------------------------
    # 1. Load dataframes + configs into a single table
    # ------------------------------------------------    
    if 'cam' not in name:
        runs = get_runs(name, filters={"config.n_maj_sources":3})
    else:
        runs = get_runs(name, filters={})
    for run_ in runs:
        df = run_.history()
        json_cfg = json.loads(run_.json_config)
        
        acq = json_cfg.get("acquisition", None)
        if acq is None:
            continue  # skip if no acquisition type
        acq = acq['value']
        # Each df contains multiple rows with "num points" and "wga test"
        if "num points" not in df.columns or "wga test" not in df.columns:
            continue

        for _, row in df.iterrows():
            all_rows.append({
                "acquisition": acq,
                "num_points": row["num points"],
                "wga test": row["wga test"]
            })

    # Convert into one DataFrame
    full_df = pd.DataFrame(all_rows)
    # ------------------------------------------------
    # 2. Group by acquisition type and num_points
    # ------------------------------------------------
    agg = (full_df
           .groupby(["acquisition", "num_points"])
           .agg(mean_wga=("wga test", "mean"),
                std_wga=("wga test", "std"))
           .reset_index())

    # ------------------------------------------------
    # 3. Plot with distinct colors (tab20 supports up to 20 colors)
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    if acquisitions is None:
        acquisitions = sorted(agg["acquisition"].unique())
    n_acq = len(acquisitions)

    # determine common plotting cutoff: compute max num_points per acquisition, then take min of those maxima
    max_per_acq = {}
    for acq in acquisitions:
        sub_max = agg.loc[agg["acquisition"] == acq, "num_points"]
        if sub_max.empty:
            max_per_acq[acq] = -1
        else:
            max_per_acq[acq] = int(sub_max.max())
    # only consider acquisitions that have data (max >= 0)
    valid_maxes = [v for v in max_per_acq.values() if v >= 0]
    if not valid_maxes:
        print("No valid num_points ranges to plot.")
        return
    common_max = min(valid_maxes)

    # assign fixed colors for specific acquisition names
    reserved_colors = {
        "random": "black",
        "uniform_sources": "orange",
        "random_gdro": "red",
        "entropy": "hotpink",
        'entropy_per_source_soft_rank': "blue",
        'n_largest_soft_rank': "green"
    }

    # map acquisition names for legend display
    acq_name_map = {
        "n_largest_soft_rank": "EntPerSourceTopM-SoftRank",
        "entropy_per_source_soft_rank": "EntPerSourceProb-SoftRank"
    }

    # build final color map: reserved names get fixed colors; others get distinct hues
    color_map = {}
    for acq in acquisitions:
        if acq in reserved_colors:
            color_map[acq] = reserved_colors[acq]

    other_acqs = [a for a in acquisitions if a not in color_map]
    n_other = len(other_acqs)
    if n_other > 0:
        if n_other <= 20:
            cmap = mpl.cm.get_cmap("tab20")
            palette = [cmap(i) for i in range(n_other)]
        else:
            cmap1 = mpl.cm.get_cmap("tab20")
            cmap2 = mpl.cm.get_cmap("tab20b")
            cmap3 = mpl.cm.get_cmap("tab20c")
            palette = ([cmap1(i) for i in range(20)] +
                       [cmap2(i) for i in range(20)] +
                       [cmap3(i) for i in range(20)])[:n_other]
        for acq, col in zip(other_acqs, palette):
            color_map[acq] = col

    for idx, acq in enumerate(acquisitions):
        # sub = agg[agg["acquisition"] == acq].sort_values("num_points")
        # restrict to the common range so all acquisitions plot to the same x (<= common_max)
        sub = agg[(agg["acquisition"] == acq) & (agg["num_points"] <= common_max)].sort_values("num_points")
 
        x = sub["num_points"].to_numpy()
        y = sub["mean_wga"].to_numpy()
        std = sub["std_wga"].to_numpy()
 
        color = color_map.get(acq, None)
        label = acq_name_map.get(acq, acq)
        ax.plot(x, y, marker="o", label=label, color=color)
        ax.fill_between(x, y - std, y + std, alpha=0.2, color=color)

    ax.set_xlabel("Num Points")
    ax.set_ylabel("WGA Test")
    ax.set_title("WGA Test vs Num Points (Mean ± Std Across Experiments), Grouped by Acquisition Type")
    ax.legend(title="Acquisition", loc='lower right')
    ax.grid(True)
    fig.tight_layout()
    
    out_path = f"wga_by_acquisition_{name}.png"
    fig.canvas.draw()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


acquisitions = ['entropy_per_source_soft_rank', 'n_largest_soft_rank', 'entropy', 'random', 'uniform_sources', 'random_gdro']

#wb
#plot_wga_by_acquisition('wb_results_jan', acquisitions=acquisitions)
#plot_test_acc_range_by_num_points('wb_results_jan', acquisitions=acquisitions)

#cmnist
#plot_wga_by_acquisition('cmnist_dino_jan', acquisitions=acquisitions)
#plot_test_acc_range_by_num_points('cmnist_dino_jan', acquisitions=acquisitions)
#cmnist_10
#plot_wga_by_acquisition('cmnist_10', acquisitions=['entropy', 'entropy_per_group_soft_rank', 'entropy_per_group_soft_max', 'n_largest_soft_max','n_largest_soft_rank', 'random', 'uniform_groups', 'random_gdro'])
#plot_test_acc_range_by_num_points('cmnist_10', acquisitions=['entropy', 'entropy_per_group_soft_rank', 'entropy_per_group_soft_max', 'n_largest_soft_max','n_largest_soft_rank', 'random', 'uniform_groups', 'random_gdro'])

#camelyon
#plot_wga_by_acquisition('cam_results_jan', acquisitions=acquisitions)
plot_test_acc_range_by_num_points('cam_results_jan', acquisitions=acquisitions)

#celeba
#plot_wga_by_acquisition('celeba_results_jan', acquisitions=acquisitions)
#plot_test_acc_range_by_num_points('celeba_results_jan  ', acquisitions=acquisitions)
