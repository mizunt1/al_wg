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

def get_dataframe(name, acquisitions=None):
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
        
        # Filter by acquisitions if provided
        if acquisitions is not None and acq not in acquisitions:
            continue
        
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
    return agg


def df_to_latex_table(df, thresholds=[60, 70, 80], acquisitions_map=None, acquisitions_order=None):
    """
    Convert a dataframe with columns [acquisition, num_points, mean_wga] 
    into a LaTeX table showing num_points needed to reach each threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns: 'acquisition', 'num_points', 'mean_wga'
    thresholds : list
        List of mean_wga thresholds to report (default: [60, 70, 80])
    acquisitions_map : dict, optional
        Dictionary mapping acquisition names to display names in the table
    acquisitions_order : list, optional
        List specifying the order of acquisitions in the table. If not provided, acquisitions are sorted alphabetically.
    
    Returns:
    --------
    str : LaTeX table code
    """
    
    if acquisitions_map is None:
        acquisitions_map = {}
    
    # Validate dataframe
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame head:\n{df.head()}")
    
    if df.empty:
        print("ERROR: DataFrame is empty!")
        return ""
    
    if 'mean_wga' not in df.columns:
        print(f"ERROR: 'mean_wga' column not found. Available columns: {df.columns.tolist()}")
        return ""
    
    # Get unique acquisitions
    available_acquisitions = set(df['acquisition'].unique())
    print(f"Acquisitions found in dataframe: {available_acquisitions}")
    print(f"Acquisitions order requested: {acquisitions_order}")
    print(f"Acquisitions map: {acquisitions_map}")
    
    # Determine order: use acquisitions_order if provided, otherwise sort alphabetically
    if acquisitions_order is not None:
        # Filter to only acquisitions that exist in the dataframe
        acquisitions = [acq for acq in acquisitions_order if acq in available_acquisitions]
    else:
        acquisitions = sorted(available_acquisitions)
    
    print(f"Final acquisition order for table: {acquisitions}")
    
    # For each acquisition, find the minimum num_points to reach each threshold
    table_data = []
    for acq in acquisitions:
        acq_df = df[df['acquisition'] == acq].sort_values('num_points')
        print(f"\n{acq}: {len(acq_df)} rows")
        print(f"  num_points range: {acq_df['num_points'].min()} - {acq_df['num_points'].max()}")
        print(f"  mean_wga range: {acq_df['mean_wga'].min():.2f} - {acq_df['mean_wga'].max():.2f}")
        
        # Apply acquisition name mapping for display (keep original if not in map)
        display_name = acquisitions_map.get(acq, acq)
        print(f"  Display name: {acq} -> {display_name}")
        row = {'Acquisition': display_name}
        
        for threshold in thresholds:
            # Convert threshold to percentage (multiply by 100)
            threshold_pct = int(threshold * 100)
            # Find first row where mean_wga >= threshold
            matching = acq_df[acq_df['mean_wga'] >= threshold]
            if not matching.empty:
                min_points = int(matching.iloc[0]['num_points'])
                row[f'{threshold_pct}%'] = min_points
                print(f"    Threshold {threshold_pct}%: {min_points} points")
            else:
                row[f'{threshold_pct}%'] = '—'  # em dash for not reached
                print(f"    Threshold {threshold_pct}%: not reached")
        
        table_data.append(row)
    
    # Build LaTeX table
    latex_lines = []
    latex_lines.append(r'\begin{table}[h]')
    latex_lines.append(r'\centering')
    
    # Create column specification
    ncols = len(thresholds) + 1  # acquisition name + thresholds
    col_spec = '|l' + '|c' * len(thresholds) + '|'
    latex_lines.append(r'\begin{tabular}{' + col_spec + '}')
    latex_lines.append(r'\hline')
    
    # Header row
    header = 'Acquisition'
    for threshold in thresholds:
        threshold_pct = int(threshold * 100)
        header += f' & {threshold_pct}\\%'
    header += r' \\ \hline'
    latex_lines.append(header)
    
    # Data rows
    for row in table_data:
        line = row['Acquisition']
        for threshold in thresholds:
            threshold_pct = int(threshold * 100)
            line += f" & {row[f'{threshold_pct}%']}"
        line += r' \\'
        latex_lines.append(line)
    
    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\caption{Number of points needed to reach each WGA threshold}')
    latex_lines.append(r'\label{tab:wga_thresholds}')
    latex_lines.append(r'\end{table}')
    
    return '\n'.join(latex_lines)


acquisitions = ['entropy_per_source_soft_rank', 'n_largest_soft_rank', 'uniform_sources', 'entropy', 'random', 'random_gdro']
acquisitions_map = {
    "n_largest_soft_rank": " \textsc{EntPerSourceProb-SoftRank}",
    "entropy_per_source_soft_rank": " \textsc{EntPerSourceTopM-SoftRank}",
    "uniform_sources": " uniform sources", 
     "random_gdro": " random g-DRO", 
}

def pull_res_print_table(wb_name, thresholds=[0.40, 0.60, 0.70, 0.80], acquisitions=acquisitions, acquisitions_map=acquisitions_map):
    df = get_dataframe(wb_name, acquisitions=acquisitions)
    latex_table = df_to_latex_table(df, thresholds=thresholds, acquisitions_map=acquisitions_map, acquisitions_order=acquisitions)
    print(latex_table)

#cmnist
#pull_res_print_table('cmnist_dino_jan2', thresholds=[0.4, 0.6, 0.7, 0.8], acquisitions=acquisitions, acquisitions_map=acquisitions_map)
#wb
pull_res_print_table('wb_results_jan', thresholds=[0.4, 0.6, 0.7, 0.8], acquisitions=acquisitions, acquisitions_map=acquisitions_map)