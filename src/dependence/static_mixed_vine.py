import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pyvinecopulib as pv
from scipy import stats

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
        allow_rotations=True,        # Enables negative correlations
        num_threads = os.cpu_count()
    )

    model = pv.Vinecop(d=N)

    # Structure Selection & Parameter Fitting (The function finds the best tree structure, families, and parameters)
    model.select(u_data, controls=controls)
    
    return model
