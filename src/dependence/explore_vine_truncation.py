import pyvinecopulib as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

def analyze_tree_subsets(model, u_train, var_names, name):
    """Analyzes the topological density and prompts the user for the cutoff."""
    d = len(var_names)
    is_spot = {i: ('PC' not in var_names[i]) for i in range(d)}
    
    model_json = model.to_json()
    M = np.array(model.matrix, dtype=np.int64)
    if M.max() == d: M = M - 1  
    if np.sum(M[0] >= 0) > np.sum(M[-1] >= 0): M = np.flipud(M)
        
    results = []
    max_tree = d - 1 
    algorithmic_cutoff = None
    
    print(f"\nAnalyzing Economic Subsets and In-Sample LL for {name} ({max_tree} trees)...")
    print(f"\n{'Tree':<6} | {'Spot-Spot':<12} | {'Density':<10} | {'Spot-Vol':<10} | {'Vol-Vol':<10}")
    print("-" * 60)
    
    for tree_idx in range(max_tree): 
        lvl = tree_idx + 1
        spot_spot, spot_vol, vol_vol = 0, 0, 0
        edges = d - 1 - tree_idx
        row = d - 1 - tree_idx
        
        for col in range(edges):
            v1_spot = is_spot[M[row, col]]
            v2_spot = is_spot[M[col, col]]
            
            if v1_spot and v2_spot: spot_spot += 1
            elif not v1_spot and not v2_spot: vol_vol += 1
            else: spot_vol += 1
                
        total_edges = spot_spot + spot_vol + vol_vol
        spot_density = (spot_spot / total_edges) if total_edges > 0 else 0.0
        
        if algorithmic_cutoff is None and spot_density <= 0.05:
            algorithmic_cutoff = lvl
                
        pct_involving_spot = ((spot_spot + spot_vol) / total_edges) * 100 if total_edges > 0 else 0
        
        trunc_model = pv.Vinecop.from_json(model_json)
        trunc_model.truncate(lvl) 
        is_ll = trunc_model.loglik(u_train)
        
        results.append({
            'Tree': lvl, 'Spot_Spot': spot_spot, 'Spot_Vol': spot_vol, 'Vol_Vol': vol_vol,
            'Pct_Involving_Spot': pct_involving_spot, 'IS_LL': is_ll
        })
        
        if lvl <= 30:
            marker = "<-- 5% CUTOFF MET" if lvl == algorithmic_cutoff else ""
            print(f"{lvl:<6} | {spot_spot:<12} | {spot_density*100:>5.1f}%    | {spot_vol:<10} | {vol_vol:<10} {marker}")

    if algorithmic_cutoff is None:
        algorithmic_cutoff = max_tree
    
    print(f"\n" + "="*50)
    print(f"The algorithm detected the 5% density terminal plateau at Tree {algorithmic_cutoff}.")
    
    user_input = input(f"Press Enter to accept Algorithm Truncation [{algorithmic_cutoff}] or type a manual override: ")
    chosen_k = int(user_input) if user_input.strip() else algorithmic_cutoff
    
    return pd.DataFrame(results), chosen_k, d

def plot_combined_decay(df_har, k_har, df_nsde, k_nsde, d, save_path):
    """Generates a single, side-by-side 1x2 figure with a unified global legend."""
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    
    # 1x2 Subplots sharing the primary Y-axis (Number of Edges)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    
    # --- PLOT HAR-GARCH ---
    b1 = ax1.bar(df_har['Tree'], df_har['Spot_Spot'], label='Spot-Spot (Contagion)', color='#3f51b5', edgecolor='white', linewidth=0.5)
    b2 = ax1.bar(df_har['Tree'], df_har['Spot_Vol'], bottom=df_har['Spot_Spot'], label='Spot-Vol (Leverage)', color='#c62828', edgecolor='white', linewidth=0.5)
    b3 = ax1.bar(df_har['Tree'], df_har['Vol_Vol'], bottom=df_har['Spot_Spot']+df_har['Spot_Vol'], label='Vol-Vol (Spillover)', color='#bdbdbd', alpha=0.7, edgecolor='white', linewidth=0.5)
    
    ax1.set_xlabel("Vine Tree Level", fontsize=14)
    ax1.set_ylabel("Number of Edges (Subset Composition)", fontsize=14)
    ax1.set_title("Panel A: HAR-GARCH Copula", fontsize=15, fontweight='bold')
    ax1.set_xticks(np.arange(1, d, 4))
    ax1.set_xlim(0, d)
    
    ax1_ll = ax1.twinx()
    l1, = ax1_ll.plot(df_har['Tree'], df_har['IS_LL'], color='darkgreen', marker='o', linewidth=2.5, markersize=4, label='In-Sample Log-Likelihood')
    v1 = ax1.axvline(x=k_har, color='black', linestyle='--', linewidth=2.5, label=r'5% Density Truncation Limit')
    
    # --- PLOT NSDE ---
    ax2.bar(df_nsde['Tree'], df_nsde['Spot_Spot'], color='#3f51b5', edgecolor='white', linewidth=0.5)
    ax2.bar(df_nsde['Tree'], df_nsde['Spot_Vol'], bottom=df_nsde['Spot_Spot'], color='#c62828', edgecolor='white', linewidth=0.5)
    ax2.bar(df_nsde['Tree'], df_nsde['Vol_Vol'], bottom=df_nsde['Spot_Spot']+df_nsde['Spot_Vol'], color='#bdbdbd', alpha=0.7, edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel("Vine Tree Level", fontsize=14)
    ax2.set_title("Panel B: Neural SDE Copula", fontsize=15, fontweight='bold')
    ax2.set_xticks(np.arange(1, d, 4))
    ax2.set_xlim(0, d)
    
    ax2_ll = ax2.twinx()
    ax2_ll.plot(df_nsde['Tree'], df_nsde['IS_LL'], color='darkgreen', marker='o', linewidth=2.5, markersize=4)
    ax2.axvline(x=k_nsde, color='black', linestyle='--', linewidth=2.5)

    # --- SYNCHRONIZE SECONDARY Y-AXIS (LOG-LIKELIHOOD) ---
    min_ll = min(df_har['IS_LL'].min(), df_nsde['IS_LL'].min())
    max_ll = max(df_har['IS_LL'].max(), df_nsde['IS_LL'].max())
    pad = (max_ll - min_ll) * 0.05
    
    ax1_ll.set_ylim(min_ll - pad, max_ll + pad)
    ax2_ll.set_ylim(min_ll - pad, max_ll + pad)
    
    # Hide the ticks and labels on the first graph's right axis to avoid middle clutter
    ax1_ll.tick_params(right=False, labelright=False)
    
    # Only label the far-right axis of the entire figure
    ax2_ll.set_ylabel("In-Sample Log-Likelihood", color='darkgreen', fontsize=14)

    # --- GLOBAL UNIFIED LEGEND ---
    handles = [b1, b2, b3, l1, v1]
    labels = [h.get_label() for h in handles]
    
    # Place legend above the entire figure
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=5, frameon=True, fontsize=12, shadow=True)
    
    # Manually adjust subplots to make room for the legend and push graphs close together
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.05, right=0.95, wspace=0.05)
    
    # Pass the legend as an extra artist so bbox_inches='tight' doesn't crop it
    fig.savefig(save_path, dpi=300, bbox_inches='tight', bbox_extra_artists=(legend,))
    print(f"\n[+] Master combined graph saved successfully to: {save_path}")

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    res_dir = os.path.join(project_root, "results", "dynamics")
    graph_dir = os.path.join(project_root, "results", "copulas", "static", "plots")
    os.makedirs(graph_dir, exist_ok=True)

    u_spot_file = os.path.join(res_dir, "NGARCH", "uniforms_ngarch_train.csv")
    u_har_file = os.path.join(res_dir, "HAR_GARCH", "uniforms_har_garch_evt_train.csv")
    u_nsde_file = os.path.join(res_dir, "NSDE", "uniforms_nsde_train.csv")

    u_spot = pd.read_csv(u_spot_file, index_col='Date', parse_dates=True)
    u_har = pd.read_csv(u_har_file, index_col='Date', parse_dates=True)
    u_nsde = pd.read_csv(u_nsde_file, index_col='Date', parse_dates=True)

    global_valid_dates = u_spot.index.intersection(u_har.index).intersection(u_nsde.index)
    u_spot, u_har, u_nsde = u_spot.loc[global_valid_dates], u_har.loc[global_valid_dates], u_nsde.loc[global_valid_dates]

    # --- 1. Analyze HAR-GARCH ---
    print(f"\n{'='*50}\n--- 1/2: Exploring Joint Copula: Spot + HAR-GARCH ---\n{'='*50}")
    combined_u_har = pd.concat([u_spot, u_har], axis=1)
    var_names_har = combined_u_har.columns.tolist()
    d_har = combined_u_har.shape[1]
    
    controls_har = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.student, 
                    pv.BicopFamily.frank, pv.BicopFamily.clayton, pv.BicopFamily.gumbel],
        selection_criterion="aic", tree_criterion="tau", allow_rotations=True,
        num_threads=os.cpu_count()-1, threshold=0.05, trunc_lvl=d_har-1
    )
    model_har = pv.Vinecop(d=d_har)
    model_har.select(combined_u_har.to_numpy(), controls=controls_har)
    df_har, k_har, d = analyze_tree_subsets(model_har, combined_u_har.to_numpy(), var_names_har, "HAR-GARCH-EVT")

    # --- 2. Analyze NSDE ---
    print(f"\n{'='*50}\n--- 2/2: Exploring Joint Copula: Spot + NSDE ---\n{'='*50}")
    combined_u_nsde = pd.concat([u_spot, u_nsde], axis=1)
    var_names_nsde = combined_u_nsde.columns.tolist()
    
    controls_nsde = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.student, 
                    pv.BicopFamily.frank, pv.BicopFamily.clayton, pv.BicopFamily.gumbel],
        selection_criterion="aic", tree_criterion="tau", allow_rotations=True,
        num_threads=os.cpu_count()-1, threshold=0.05, trunc_lvl=d_har-1
    )
    model_nsde = pv.Vinecop(d=d_har)
    model_nsde.select(combined_u_nsde.to_numpy(), controls=controls_nsde)
    df_nsde, k_nsde, _ = analyze_tree_subsets(model_nsde, combined_u_nsde.to_numpy(), var_names_nsde, "NSDE")

    # --- 3. Plot Combined Figure ---
    print("\nGenerating final Master Figure...")
    save_path = os.path.join(graph_dir, "Decay_Combined_Master.png")
    plot_combined_decay(df_har, k_har, df_nsde, k_nsde, d, save_path)

    print("\n" + "="*50)
    print("PHASE 1 COMPLETE. UPDATE YOUR config/settings.py WITH:")
    print(f"K_HAR_GARCH = {k_har}")
    print(f"K_NSDE = {k_nsde}")
    print("="*50)