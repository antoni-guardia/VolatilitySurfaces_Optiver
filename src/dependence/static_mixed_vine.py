import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pyvinecopulib as pv

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
            pv.BicopFamily.joe,      # Sharp Upper Tail (Booms)
            pv.BicopFamily.bb1,      # Asymmetric (Crash + Boom)
            pv.BicopFamily.bb7,      # Asymmetric Alternative
            pv.BicopFamily.bb8       # Asymmetric Alternative
        ],
        selection_criterion="bic",
        trunc_lvl=N-1,               # Fit full tree
        tree_criterion="rho",        # Use correlation to order the tree 
        allow_rotations=True         # Enables negative correlations
    )

    model = pv.Vinecop(d=N)

    # Structure Selection & Parameter Fitting (The function finds the best tree structure, families, and parameters)
    model.select(u_data, controls=controls)
    
    return model

# Visualization of the Vine Structure
def plot_vine_structure(model, asset_names, tree_index=0):
    structure_matrix = model.matrix
    d = len(asset_names)
    
    G = nx.Graph()
    for name in asset_names:
        G.add_node(name)
    
    edge_list = []
    M = np.array(structure_matrix)
    
    row_idx = d - 1 - tree_index
    for i in range(d - 1 - tree_index):
        source = int(M[row_idx, i]) - 1 
        target = int(M[i, i]) - 1

        try:
            tau = model.pair_copulas[tree_index][i].tau
        except:
            tau = 0
            
        edge_list.append((asset_names[source], asset_names[target], tau))

    for u, v, w in edge_list:
        G.add_edge(u, v, weight=w)

    # Plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw edges with varying thickness based on Tau
    weights = [abs(x[2]) * 5 for x in edge_list]
    edge_colors = ['green' if x[2] > 0 else 'red' for x in edge_list]
    
    nx.draw_networkx_edges(G, pos, width=weights, edge_color=edge_colors, alpha=0.7)
    
    # Add edge labels (Tau values)
    edge_labels = {(u,v): f"{w:.2f}" for u,v,w in edge_list}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"R-Vine Tree {tree_index+1} Structure (Width = Dependence Strength)", fontsize=16)
    plt.axis('off')
    plt.show()

# Visualization of the Copula Families in the First Tree
def plot_family_heatmap(model, asset_names):
    d = len(asset_names)
    M = np.array(model.matrix)
    
    fam_matrix = np.full((d, d), np.nan)
    labels_matrix = np.full((d, d), "", dtype=object)
    
    # Families enum to string
    fam_map = {
        pv.BicopFamily.gaussian: 1, "Gaussian": 1,
        pv.BicopFamily.student: 2, "Student": 2,
        pv.BicopFamily.clayton: 3, "Clayton": 3,
        pv.BicopFamily.gumbel: 4, "Gumbel": 4,
        pv.BicopFamily.frank: 5, "Frank": 5,
        pv.BicopFamily.joe: 6, "Joe": 6,
        pv.BicopFamily.bb1: 7, "BB1": 7,
        pv.BicopFamily.bb7: 8, "BB7": 8,
        pv.BicopFamily.bb8: 9, "BB8": 9,
        pv.BicopFamily.indep: 0, "Indep": 0,
    }
    
    # Fill for Tree 0
    tree = 0
    row_idx = d - 1 - tree
    for i in range(d - 1):
        source = int(M[row_idx, i]) - 1 
        target = int(M[i, i]) - 1
        
        # Get Family
        pc = model.pair_copulas[tree][i]
        fam_name = str(pc.family).split('.')[-1]
        
        # Map to integer for coloring
        val = fam_map.get(fam_name, -1)
        if "bb" in fam_name: val = 7 # Group BBs together for simpler coloring
        
        fam_matrix[source, target] = val
        fam_matrix[target, source] = val
        labels_matrix[source, target] = fam_name
        labels_matrix[target, source] = fam_name

    plt.figure(figsize=(10, 8))
    
    # Custom colormap
    cmap = sns.color_palette("Set3", 10)
    sns.heatmap(fam_matrix, annot=labels_matrix, fmt="", cmap=cmap, 
                xticklabels=asset_names, yticklabels=asset_names, 
                cbar=False, linewidths=1, linecolor='gray')
    
    plt.title("Copula Families in First Tree (Direct Connections)", fontsize=16)
    plt.show()
