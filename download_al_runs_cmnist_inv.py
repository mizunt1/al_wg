import wandb
import json
from collections import defaultdict
import os
import scipy
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def get_runs(name, filters):
    api = wandb.Api(timeout=19)
    return api.runs(name, filters=filters)


def plot_wga_by_acquisition(name, acquisitions=None, group_key='num_digits_per_target', group_values=None):
    """
    Plot WGA grouped by acquisition and a chosen group_key.
    - `acquisitions`: optional list of acquisition names to plot (default = all).
    - `group_values`: optional list of group_key values to plot (default = all).
    """
    # map acquisition names (rename early in the pipeline)
    acq_name_map = {
        "n_largest_soft_rank": "m_largest_sources_soft_rank"
    }

    all_rows = []

    # ------------------------------------------------
    # 1. Load dataframes + configs into a single table
    # ------------------------------------------------
    runs = get_runs(name, filters={})
    for run_ in runs:
        df = run_.history()
        json_cfg = json.loads(run_.json_config)

        acq_entry = json_cfg.get("acquisition", None)
        acq = acq_entry.get("value") if isinstance(acq_entry, dict) else acq_entry
        
        # apply acquisition name mapping
        acq = acq_name_map.get(acq, acq)

        group_entry = json_cfg.get(group_key, None)
        group_val = group_entry.get("value") if isinstance(group_entry, dict) else group_entry

        if acq is None or group_val is None:
            continue
        if "num points" not in df.columns or "wga" not in df.columns:
            continue

        for _, row in df.iterrows():
            all_rows.append({
                "acquisition": acq,
                group_key: group_val,
                "num_points": row["num points"],
                "wga": row["wga"]
            })

    full_df = pd.DataFrame(all_rows)
    if full_df.empty:
        print("No data collected from runs (empty dataframe).")
        return

    # ------------------------------------------------
    # 2. Group by acquisition, group_key, num_points
    # ------------------------------------------------
    agg = (full_df
           .groupby(["acquisition", group_key, "num_points"])
           .agg(mean_wga=("wga", "mean"),
                std_wga=("wga", "std"))
           .reset_index())

    # determine available group values and which to plot
    available_group_values = sorted(agg[group_key].unique())
    if group_values is None:
        group_values_to_plot = available_group_values
    else:
        group_values_to_plot = [v for v in group_values if v in available_group_values]
        if not group_values_to_plot:
            print("None of the requested group_key values were found in the data.")
            return

    # determine available acquisitions restricted to chosen group_values
    available_acqs = sorted(agg.loc[agg[group_key].isin(group_values_to_plot), "acquisition"].unique())
    if acquisitions is None:
        acquisitions_to_plot = available_acqs
    else:
        acquisitions_to_plot = [a for a in acquisitions if a in available_acqs]
        if not acquisitions_to_plot:
            print("None of the requested acquisitions were found in the data (for the chosen group values).")
            return

    # ------------------------------------------------
    # 3. Plot (one line per (acquisition, group_key))
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    n_groups = max(1, len(group_values_to_plot))

    # assign fixed colors for specific acquisition names
    reserved_colors = {
        "random": "red",
        "uniform_sources": "orange",
        "random_gdro": "yellow",
        "entropy": "hotpink",
        "entropy_per_source_soft_rank": "blue",
        "m_largest_sources_soft_rank": "green",
    }

    # assign line styles for each group_key value
    line_styles = ['-', '--', '-.', ':']
    group_line_styles = {gv: line_styles[i % len(line_styles)] for i, gv in enumerate(group_values_to_plot)}

    color_map_for_group = {}
    for group_val in group_values_to_plot:
        acqs = sorted(agg.loc[
            (agg[group_key] == group_val) & (agg["acquisition"].isin(acquisitions_to_plot)),
            "acquisition"
        ].unique())

        acq_to_color = {}
        # assign only reserved colors
        for acq in acqs:
            if acq in reserved_colors:
                acq_to_color[acq] = reserved_colors[acq]

        color_map_for_group[group_val] = acq_to_color

    # compute common cutoff: for each (acq, group_val) that will be plotted, get its max num_points
    max_per_line = {}
    for group_val in group_values_to_plot:
        for acq in color_map_for_group.get(group_val, {}).keys():
            sub_all = agg[(agg["acquisition"] == acq) & (agg[group_key] == group_val)]
            if not sub_all.empty:
                max_per_line[(acq, group_val)] = int(sub_all["num_points"].max())

    if not max_per_line:
        print("No data to plot after filtering.")
        return

    common_max = min(max_per_line.values())

    # plotting lines (trimmed to common_max so all lines equal length)
    for group_val in group_values_to_plot:
        acqs = sorted(color_map_for_group.get(group_val, {}).keys())
        line_style = group_line_styles[group_val]
        for acq in acqs:
            sub = (agg[(agg["acquisition"] == acq) & (agg[group_key] == group_val)]
                   .loc[lambda d: d["num_points"] <= common_max]
                   .sort_values("num_points"))
            if sub.empty:
                continue
            x = sub["num_points"].to_numpy()
            y = sub["mean_wga"].to_numpy()
            std = sub["std_wga"].to_numpy()
            color = color_map_for_group[group_val][acq]
            label = f"{acq} ({group_key}={group_val})"
            ax.plot(x, y, marker="o", label=label, color=color, linestyle=line_style)
            ax.fill_between(x, y - std, y + std, alpha=0.2, color=color)

    ax.set_xlabel("Num Points")
    ax.set_ylabel("WGA")
    ax.set_title(f"WGA vs Num Points (Mean ± Std), Grouped by {group_key}")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax.grid(True)
    fig.tight_layout()

    out_path = f"wga_by_acquisition_{name}_grouped_by_{group_key}.png"
    fig.canvas.draw()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
 

plot_wga_by_acquisition('cmnist_num_sources', group_key='n_maj_sources', group_values=[2, 6])
#plot_wga_by_acquisition('cmnist_easy', group_key='num_digits_per_target', group_values=[1, 5])

#plot_wga_by_acquisition('cmnist_spurious_inv', group_key='spurious_noise', group_values=[0.005, 0.1, 0.5],
                        #acquisitions=['entropy_per_group_soft_max', 'uniform_groups'])
#plot_wga_by_acquisition('celeba_res_nov')
