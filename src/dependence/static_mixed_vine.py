import pyvinecopulib as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Fits a Static Mixed R-Vine Copula to the data
def fit_static_mixed_vine(u_data):
    T, N = u_data.shape
    
    # Define the Fit Controls
    controls = pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.indep,    # Sparsity (No dependence)
            pv.BicopFamily.gaussian, # Standard Correlation
            pv.BicopFamily.student,  # Symmetric Tail Dependence
            pv.BicopFamily.frank,    # Symmetric Body Dependence (No tails)
            pv.BicopFamily.clayton,  # Lower Tail (Crashes)
            pv.BicopFamily.gumbel,   # Upper Tail (Rallies)
            #pv.BicopFamily.bb1,      # Clayton-Gumbel Mix
            #pv.BicopFamily.bb7       # Joe-Clayton Mix
        ],
        selection_criterion="aic",
        trunc_lvl=N-1,               # Fit full tree
        tree_criterion="tau",        # Use correlation to order the tree 
        allow_rotations=True,        # Enables negative correlations
        num_threads=os.cpu_count()-1
    )

    model = pv.Vinecop(d=N)

    # Structure Selection & Parameter Fitting (The function finds the best tree structure, families, and parameters)
    model.select(u_data, controls=controls)
    
    return model

# Visualizations
def plot_large_heatmap(u_df, name, save_path):
    """Saves a massive, high-res heatmap to identify dependency clusters."""
    plt.figure(figsize=(25, 20))
    tau_matrix = u_df.corr(method='kendall')
    # annot=False is critical here to prevent the 'black scribble'
    sns.heatmap(tau_matrix, annot=False, cmap='coolwarm', center=0, cbar_kws={'label': "Kendall's Tau"})
    plt.title(f"Joint Kendall's Tau Heatmap: {name}", fontsize=25)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_family_dist(model, name, save_path):
    """Visualizes which copula families (Clayton/Gumbel) dominate the system."""
    families = []
    for tree_cops in model.pair_copulas:
        for bicop in tree_cops:
            families.append(str(bicop.family).split('.')[-1].capitalize())
    
    plt.figure(figsize=(12, 7))
    pd.Series(families).value_counts().plot(kind='bar', color='teal')
    plt.title(f"Selected Copula Families: {name}", fontsize=18)
    plt.ylabel("Frequency (Edges in Vine)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_spot_vol_scatters(u_df, name, save_path):
    """Specifically plots the Top 6 Spot vs. Vol-Factor relationships."""
    # Find columns starting with 'G_PC' or 'L_PC' (Factors) and Spot assets
    factors = [c for c in u_df.columns if '_PC_' in c]
    spots = [c for c in u_df.columns if '_PC_' not in c]
    
    # Simple logic: Find the factor with the highest absolute correlation to the first Spot
    target_spot = spots[0] # Usually 'SPY' or your main asset
    corrs = u_df[factors].corrwith(u_df[target_spot], method='kendall').abs().sort_values(ascending=False)
    top_6_factors = corrs.index[:6]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Top Spot-Vol Dependencies: {name} (Target: {target_spot})", fontsize=20)
    
    for i, factor in enumerate(top_6_factors):
        ax = axes[i // 3, i % 3]
        ax.scatter(u_df[target_spot], u_df[factor], s=1, alpha=0.3, color='darkblue')
        ax.set_title(f"{target_spot} vs {factor}\nTau: {corrs[factor]:.2f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    res_dir = os.path.join(project_root, "outputs", "dynamics")
    out_dir = os.path.join(project_root, "outputs", "copulas")
    os.makedirs(out_dir, exist_ok=True)
    
    u_spot_file = os.path.join(res_dir, "uniforms_ngarch_t.csv")
    u_spot = pd.read_csv(u_spot_file, index_col=0)
    u_spot.index = pd.to_datetime(u_spot.index).date

    # Align Dates
    u_har_path = os.path.join(res_dir, "uniforms_har_garch_evt.csv")
    u_nsde_path = os.path.join(res_dir, "nsde_uniforms.csv")

    if os.path.exists(u_har_path) and os.path.exists(u_nsde_path):
        u_har_ref = pd.read_csv(u_har_path, index_col=0)
        u_har_ref.index = pd.to_datetime(u_har_ref.index).date
        u_nsde_ref = pd.read_csv(u_nsde_path, index_col=False)
        
        valid_har_dates = u_spot.index.intersection(u_har_ref.index)
        global_valid_dates = valid_har_dates[-len(u_nsde_ref):]
        print(f"Evaluation Period: {global_valid_dates[0]} to {global_valid_dates[-1]}")

    else:
        print("Missing required files to establish global dates.")
        exit()

    # Fit Models 
    factor_sets = {"HAR-GARCH-EVT": "uniforms_har_garch_evt.csv", "NSDE": "nsde_uniforms.csv"}
    results_summary = []

    for factor_name, file_name in factor_sets.items():
        u_factor_path = os.path.join(res_dir, file_name)
        
        if os.path.exists(u_factor_path):
            print(f"\n--- Fitting Joint Copula: Spot (NGARCH-t) + Factors ({factor_name}) ---")
            
            if factor_name == "NSDE":
                u_factors = pd.read_csv(u_factor_path, index_col=False) 
                u_factors.index = global_valid_dates
            else:
                u_factors = pd.read_csv(u_factor_path, index_col=0)
                u_factors.index = pd.to_datetime(u_factors.index).date

            combined_u = pd.concat([u_spot, u_factors], axis=1, join='inner').dropna()
            combined_u = combined_u.loc[global_valid_dates] 

            if combined_u.empty:
                print(f"ERROR: The combined dataset is empty! Could not align dates.")
                continue
            
            joint_model = fit_static_mixed_vine(combined_u.to_numpy())

            order = joint_model.order
            ordered_names = [combined_u.columns[int(i) - 1] for i in order]
            print(f"Top 5 Root Nodes (Importance): {ordered_names[:5]}")
            print(f"Log-Likelihood: {joint_model.loglik(combined_u.to_numpy()):.2f}")
            print(f"AIC: {joint_model.aic(combined_u.to_numpy()):.2f}")
            
            # Diagnostics

            save_prefix = f"joint_vine_spot_{factor_name.lower().replace('-', '_')}"
            plot_large_heatmap(combined_u, factor_name, os.path.join(out_dir, f"{save_prefix}_heatmap.png"))
            plot_family_dist(joint_model, factor_name, os.path.join(out_dir, f"{save_prefix}_families.png"))
            plot_spot_vol_scatters(combined_u, factor_name, os.path.join(out_dir, f"{save_prefix}_scatters.png"))

            with open(os.path.join(out_dir, f"{save_prefix}.json"), "w") as f:
                f.write(joint_model.to_json())
    
        else:
            print(f"Warning: {file_name} not found. Skipping {factor_name} loop.")
