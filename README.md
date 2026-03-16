# Leverage and Contagion Effects in Implied Volatility Surfaces: A Mixed Neural Vine Copula Approach

**Erasmus University Rotterdam | Erasmus School of Economics** **Partner Firm:** Optiver (Group A)  
**Authors:** Luca Leimbeck Del Duca, Melvin Bazeille, Antoni Guàrdia Sanz, Ángel Rodríguez Fernández  
**Date:** March 2026  

---

## 📌 Project Overview

Forecasting the joint evolution of implied volatility surfaces (IVS) across a high-dimensional universe of assets is a critical challenge for market makers, particularly during periods of systemic stress. Traditional econometric models often struggle to reconcile the arbitrage-free geometry of the surface with the complex, non-linear dependence structure of asset returns. 

This repository contains the codebase for a unified **Deep Generative Framework** that decomposes the joint forecasting problem into three sequential stages:
1. **Geometry:** Surface Stochastic Volatility Inspired (SSVI) parameterization and Beta-Adjusted Multilevel Functional Principal Component Analysis ($\beta^{\text{adj}}$-mfPCA) construct an arbitrage-free, low-dimensional market representation.
2. **Dynamics:** Neural Stochastic Differential Equations (Neural SDEs) model the temporal evolution of the extracted factors, capturing path-dependent memory structures.
3. **Topology:** High-dimensional, asymmetric dependence structures are estimated via a novel Differentiable Mixed Regular Vine Copula (Neural Vine and GAS Vine).

The framework explicitly quantifies the economic value of non-linear contagion modeling by eliminating "phantom diversification" and isolating systematic vanna bleed in delta-hedged risk reversals (Portfolio 1) and relative-value dispersion trades (Portfolio 2).

---

## 📂 Repository Structure

The codebase is highly modular and organized within the `src/` directory to mirror the econometric pipeline of the research paper.

```text
├── data/                               # Raw and processed Parquet/CSV data (Not tracked in Git)
├── results/                            # Output directory for fitted models, logs, and simulation CSVs
└── src/
    ├── preprocessing/                  # Data cleaning and return/rate calculations
    │   ├── preprocessing.py
    │   ├── returns.ipynb
    │   └── rf_rate.ipynb
    ├── fitting/                        # Stage 1a: Arbitrage-free SSVI calibration
    │   ├── ssvi.py                     # Core SSVI optimization logic
    │   └── main.py                     # Parallelized fitting orchestrator
    ├── compression/                    # Stage 1b: Dimensionality reduction
    │   ├── beta_multilevel_fanova.py   # Global and local functional PCA
    │   └── helpers/                    # Reconstruction and visualization tools
    ├── dynamics/                       # Stage 2: Time-series modeling (Marginals)
    │   ├── HAR_GARCH.py                # Classical autoregressive baseline
    │   ├── NeuralSDE.py                # Continuous-time deep generative model
    │   ├── NGARCH_T.py
    │   └── EVT.py                      # Extreme Value Theory tail calibration
    ├── dependence/                     # Stage 3: Joint dependence (Topology)
    │   ├── static_mixed_vine.py        # Baseline static copula (M0)
    │   ├── gas_mixed_vine.py           # Score-driven dynamic copula (M1, M3)
    │   ├── neural_mixed_vine.py        # GRU-driven differentiable copula (M2, M4)
    │   └── explore_vine_truncation.py  # D-vine truncation algorithms
    ├── backtesting/                    # Stage 4: Economic evaluation & Ex-Ante Simulations
    │   ├── portfolio1.py               # 25-Delta Risk Reversal (Directional tail risk)
    │   ├── portfolio2.py               # QQQ Vega-Neutral Dispersion (Correlation premium)
    │   ├── p1_backtest.py              # Ex-post 1-day rolling backtests
    │   ├── p2_backtest.py
    │   ├── p1_sim.py                   # Ex-ante 20-day rolling hedging-error simulation
    │   ├── p2_sim.py                   
    │   └── utils/                      # Fast vectorized Black-Scholes Greeks & Generators
    └── intermediate_diagnostics/       # Econometric validation and time-series plotting
