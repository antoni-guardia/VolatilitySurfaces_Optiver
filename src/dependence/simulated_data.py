import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

def generate_optiver_proxy_sample(T=2000, degrees_of_freedom=4.0):
    """
    Simulates a proxy sample approximating the output of the 
    SVI -> MFPCA -> HAR-GARCH-EVT-PIT pipeline.
    
    Generates Uniform(0,1) residuals with:
    1. Realistic Sector Correlations (Tech cluster, Market factor, Hedging assets).
    2. Heavy Tails (Student-t copula) to mimic financial shocks.
    
    Args:
        T (int): Number of time steps (observations).
        degrees_of_freedom (float): Controls tail heaviness (lower = more crashes).
                                    4.0 is typical for financial returns.
    """
    
    # 1. Define Asset List & Sectors
    assets = ['GLD', 'NVDA', 'AMZN', 'JPM', 'QQQ', 'MSFT', 'AAPL', 'IWM', 'SPY', 'TSLA', 'V', 'NFLX', 'TLT', 'GOOGL']
    n = len(assets)
    
    # Map assets to indices for correlation construction
    asset_map = {name: i for i, name in enumerate(assets)}
    
    # 2. Construct a Realistic Correlation Matrix
    # We start with a base correlation (Market Mode)
    rho = np.full((n, n), 0.4) # Base market correlation
    np.fill_diagonal(rho, 1.0)
    
    # Sector: Big Tech (High Beta, tight clustering)
    tech = ['NVDA', 'AMZN', 'QQQ', 'MSFT', 'AAPL', 'TSLA', 'NFLX', 'GOOGL']
    tech_idx = [asset_map[a] for a in tech]
    for i in tech_idx:
        for j in tech_idx:
            if i != j: rho[i, j] = 0.75  # Tech moves together
            
    # Sector: Core Indices (SPY, QQQ, IWM)
    indices = ['SPY', 'QQQ', 'IWM']
    idx_idx = [asset_map[a] for a in indices]
    for i in idx_idx:
        for j in idx_idx:
            if i != j: rho[i, j] = 0.85 # Indices highly correlated

    # Specific Pair: NVDA & TSLA (Momentum/Volatile)
    rho[asset_map['NVDA'], asset_map['TSLA']] = 0.65

    # Sector: Financials/Cyclicals (JPM, V, IWM, SPY)
    fin = ['JPM', 'V', 'IWM', 'SPY']
    fin_idx = [asset_map[a] for a in fin]
    for i in fin_idx:
        for j in fin_idx:
            if i != j: rho[i, j] = 0.60
            
    # Sector: Macro/Defensive (TLT, GLD) - The Diversifiers
    # These often have low or negative correlation to Equities
    macro = ['TLT', 'GLD']
    macro_idx = [asset_map[a] for a in macro]
    
    # Decorrelate Macro from Equities
    equity_indices = [i for i in range(n) if i not in macro_idx]
    for m in macro_idx:
        for e in equity_indices:
            rho[m, e] = -0.15 # Slight negative correlation (Flight to safety)
            rho[e, m] = -0.15
            
    # GLD vs TLT (often weakly positive)
    rho[asset_map['GLD'], asset_map['TLT']] = 0.25
    rho[asset_map['TLT'], asset_map['GLD']] = 0.25

    # 3. Simulate Multivariate Student-t Data
    # Robust method using Eigenvalue Decomposition (works even for non-PD matrices)
    
    # Compute Eigenvalues and Eigenvectors
    evals, evecs = np.linalg.eigh(rho)
    
    # Fix negative eigenvalues (make matrix Positive Definite)
    evals[evals < 1e-6] = 1e-6
    
    # Reconstruct the "nearest" valid covariance matrix
    # We use this square root for simulation: sqrt(Sigma) = E * sqrt(Lambda)
    B = evecs @ np.diag(np.sqrt(evals))
    
    # Generate Normal shocks: Z ~ N(0, I)
    Z = np.random.normal(size=(T, n))
    
    # Correlate shocks: X = Z @ B.T
    X_corr = Z @ B.T
    
    # Rescale columns so they have standard deviation = 1 (Required for t-distribution)
    # This fixes the "diagonal != 1" issue implicitly
    X_corr = X_corr / np.std(X_corr, axis=0, keepdims=True)
    
    # Generate Chi-Square for t-distribution mixing (Tail simulation)
    chi2_val = np.random.chisquare(df=degrees_of_freedom, size=(T, 1))
    scale = np.sqrt(degrees_of_freedom / chi2_val)
    X_t = X_corr * scale
    
    # 4. Probability Integral Transform (PIT)
    # Convert t-distributed returns -> Uniform(0,1)
    # This mimics the output of the "GARCH-EVT-PIT" step
    U_sim = student_t.cdf(X_t, df=degrees_of_freedom)
    
    return U_sim, assets