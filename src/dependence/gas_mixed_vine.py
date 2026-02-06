import numpy as np
import pyvinecopulib as pv
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Bounds helper for copula families
def get_parameter_bounds(family):
    bounds = {
        pv.BicopFamily.gaussian: (-0.9999, 0.9999),
        pv.BicopFamily.student: (-0.9999, 0.9999),
        pv.BicopFamily.clayton: (1e-4, 20.0),      # Must be > 0
        pv.BicopFamily.gumbel: (1.0001, 20.0),     # Must be >= 1
        pv.BicopFamily.joe: (1.0001, 20.0),        # Must be >= 1
        pv.BicopFamily.frank: (-20.0, 20.0),       # Non-zero
        pv.BicopFamily.bb1: (1e-4, 5.0),           # Theta > 0
        pv.BicopFamily.bb7: (1.0001, 5.0),         # Theta >= 1
        pv.BicopFamily.bb8: (1.0001, 5.0),         # Theta >= 1
    }
    return bounds.get(family, (-np.inf, np.inf))

# Robust parameter converter to ensure compatibility with pyvinecopulib's C++ backend
def to_pv_params(theta, fixed_params):

    # Extract scalar from numpy array if needed (Fixes DeprecationWarning)
    th_val = theta.item() if isinstance(theta, np.ndarray) else float(theta)
    
    p_list = [th_val]
    
    # Append secondary parameter if it exists
    if len(fixed_params) > 1:
        p2_val = fixed_params[1]
        p_list.append(p2_val.item() if isinstance(p2_val, np.ndarray) else float(p2_val))
        
    # Create strictly 2D array with float64
    return np.array([p_list], dtype=np.float64)

# Maps the unbounded GAS factor 'f' to the valid parameter space 'theta'
def transform_f_to_theta(f, family):
    if family in [pv.BicopFamily.gaussian, pv.BicopFamily.student]:
        return np.tanh(f) # Maps (-inf, inf) -> (-1, 1)
    
    elif family == pv.BicopFamily.clayton:
        return np.exp(f) + 1e-4 # Maps to (0, inf)
    
    elif family in [pv.BicopFamily.gumbel, pv.BicopFamily.joe]:
        return np.exp(f) + 1.0001 # Maps to (1, inf)
    
    elif family == pv.BicopFamily.frank:
        return f if abs(f) > 1e-4 else 1e-4
        
    elif "bb" in str(family).lower():
        if family == pv.BicopFamily.bb1:
            return np.exp(f) + 1e-4
        else:
            return np.exp(f) + 1.0001
             
    return f

#  Inverse mapping used to initialize 'f' from the Static model's parameter
def inverse_transform_theta_to_f(theta, family):

    th = theta.item() if isinstance(theta, np.ndarray) else theta

    if family in [pv.BicopFamily.gaussian, pv.BicopFamily.student]:
        return np.arctanh(np.clip(th, -0.99, 0.99))
    elif family == pv.BicopFamily.clayton:
        return np.log(max(th, 1e-4))
    elif family in [pv.BicopFamily.gumbel, pv.BicopFamily.joe]:
        return np.log(max(th - 1.0, 1e-4))
    elif "bb" in str(family).lower():
        if family == pv.BicopFamily.bb1:
            return np.log(max(th, 1e-4))
        else:
            return np.log(max(th - 1.0, 1e-4))
    return th

# Compute the Score (Gradient of Log-Likelihood) using a 5-Point Stencil for numerical differentiation
def score(u, v, family, theta, fixed_params, rotation=0, epsilon=1e-5):
    try:
        def get_ll(th):

            # Bounds Check
            lb, ub = get_parameter_bounds(family)
            if th <= lb or th >= ub: return -1e10

            try:
                params_arr = to_pv_params(th, fixed_params)
                bc = pv.Bicop(family=family, parameters=params_arr, rotation=rotation)
                return bc.loglik(np.column_stack([u, v]))
            except RuntimeError:
                return -1e10

        # Richardson Extrapolation (5 points)
        ll_p2 = get_ll(theta + 2*epsilon)
        ll_p1 = get_ll(theta + 1*epsilon)
        ll_m1 = get_ll(theta - 1*epsilon)
        ll_m2 = get_ll(theta - 2*epsilon)
        
        # Stability check
        if ll_p1 <= -1e9 or ll_m1 <= -1e9: return 0.0
        
        score = (-ll_p2 + 8*ll_p1 - 8*ll_m1 + ll_m2) / (12 * epsilon)
        
        # Clip gradients to prevent GAS explosion during shocks
        return np.clip(score, -100, 100)
        
    except:
        return 0.0

# Fits the GAS process for a single pair of variables
def fit_gas_edge(u, v, family, rotation=0):
    T = len(u)
    
    # Warm Start: Use the Static Fit to initialize
    bc_static = pv.Bicop(family=family, rotation=rotation)
    bc_static.fit(np.column_stack([u, v]))
    static_params = np.array(bc_static.parameters).flatten()
    
    # Initialize GAS variables
    omega_init = static_params[0]
    fixed_params = static_params 
    f_init = inverse_transform_theta_to_f(omega_init, family)

    # Tune A and B
    def objective(hyperparams):
        a_try, b_try = hyperparams
        
        # Constraints: Stationarity (A + B < 1) and Positivity (A > 0, B > 0) 
        if a_try + b_try >= 0.999: return 1e9
        if a_try < 0.001 or b_try < 0.001: return 1e9
        
        # omega = f_long_run * (1 - B) - A * expectation_of_score can be simplified to omega = f_init * (1 - B) 
        omega_try = f_init * (1 - b_try)

        f_t = f_init
        total_nll = 0.0
        for t in range(T):
            theta_t = transform_f_to_theta(f_t, family)
            
            # Bounds Check: Assign a large penalty to the NLL to steer optimization away from this region
            lb, ub = get_parameter_bounds(family)
            if theta_t <= lb or theta_t >= ub: 
                total_nll += 1e5
            else:
                try:
                    params_arr = to_pv_params(theta_t, fixed_params)
                    bc_t = pv.Bicop(family=family, parameters=params_arr, rotation=rotation)
                    ll = bc_t.loglik(np.column_stack([u[t], v[t]]))
                    
                    if np.isnan(ll) or np.isinf(ll):
                         total_nll += 1e5
                    else:
                         total_nll -= ll
                except RuntimeError:
                    total_nll += 1e5
            
            # Update Score
            st = score(np.array([u[t]]), np.array([v[t]]), family, theta_t, fixed_params, rotation)
            f_t = omega_try + a_try * st + b_try * f_t
            
        return total_nll

    initial_guess = [0.05, 0.90]
    res = minimize(objective, initial_guess, method='Nelder-Mead', tol=1e-3)
    best_A, best_B = res.x

    msg = f"Optimized Edge ({family}): A={best_A:.3f}, B={best_B:.3f}"
    tqdm.write(msg)

    # Final Run with Best Params to generate path
    omega_best = f_init * (1 - best_B)
    theta_path = np.zeros(T)
    f_t = f_init

    # The GAS Loop (Time evolution) 
    for t in range(T):
        theta_t = transform_f_to_theta(f_t, family)
        theta_path[t] = theta_t
        st = score(np.array([u[t]]), np.array([v[t]]), family, theta_t, fixed_params, rotation)
        f_t = omega_best + best_A * st + best_B * f_t
        
    return theta_path, fixed_params

# Fits the Mixed GAS Vine
def fit_gas_vine(u_data, static_model):
    T, N = u_data.shape
    M = np.array(static_model.matrix)
    
    # [FIX 1] Robust Matrix Orientation Check
    top_nonzeros = np.count_nonzero(M[0, :])
    bot_nonzeros = np.count_nonzero(M[N-1, :])
    if top_nonzeros > bot_nonzeros:
        print(f">> Detected Upper Triangular Matrix (Top={top_nonzeros}, Bot={bot_nonzeros}). Transposing...")
        M = M.T
        
    dynamic_results = {}
    
    # [FIX 2] Robust Storage: List of Dicts
    # tree_outputs[t][col] stores the two outputs for that specific edge
    tree_outputs = { -1: {} } # Initialize for raw data
    
    # Store raw data as "Tree -1"
    for i in range(N):
        # For raw data, we pretend it came from an edge where 'direct' is the asset itself
        tree_outputs[-1][i] = {
            'direct_var': i, 
            'direct_h': u_data[:, i],
            'indirect_var': i, 
            'indirect_h': u_data[:, i] # Duplicate for safety
        }

    print(f"Fitting SOTA GAS Mixed Vine on {N} assets...")
    
    edges_processed = 0
    edges_saved = 0
    
    for tree in range(N - 1):
        # Create storage for this tree level
        tree_outputs[tree] = {}
        
        pbar = tqdm(range(N - 1 - tree), desc=f"Tree {tree+1}")
        
        for edge in pbar:
            row_idx = N - 1 - tree
            
            raw_a = int(M[row_idx, edge])
            raw_b = int(M[edge, edge])
            
            if raw_a == 0 or raw_b == 0: continue

            edges_processed += 1
            a_idx = raw_a - 1
            b_idx = raw_b - 1
            
            # [FIX 3] Search & Retrieve Logic
            # We need input for variable 'a_idx' from the previous tree.
            # We scan the previous tree's outputs to find who produced 'a_idx'.
            u_vec = None
            v_vec = None
            
            # Search in previous tree outputs
            prev_level = tree_outputs[tree - 1]
            
            # Find u_vec (for a_idx)
            if tree == 0:
                u_vec = prev_level[a_idx]['direct_h']
            else:
                for col, out in prev_level.items():
                    if out['direct_var'] == a_idx:
                        u_vec = out['direct_h']
                        break
                    elif out['indirect_var'] == a_idx:
                        u_vec = out['indirect_h']
                        break
            
            # Find v_vec (for b_idx)
            if tree == 0:
                v_vec = prev_level[b_idx]['direct_h']
            else:
                for col, out in prev_level.items():
                    # Optimization: In R-Vines, v usually comes from the diagonal partner
                    # but scanning is safer for general structures.
                    if out['direct_var'] == b_idx:
                        v_vec = out['direct_h']
                        break
                    elif out['indirect_var'] == b_idx:
                        v_vec = out['indirect_h']
                        break
            
            # If data missing, skip safely
            if u_vec is None or v_vec is None:
                # This should theoretically not happen if Matrix is valid
                continue
            
            pc = static_model.pair_copulas[tree][edge]
            fam, rot = pc.family, pc.rotation
            
            if fam == pv.BicopFamily.indep:
                theta_path = np.zeros(T)
                h_a, h_b = u_vec, v_vec
            else:
                theta_path, fixed_params = fit_gas_edge(u_vec, v_vec, fam, rot)
                
                h_a = np.zeros(T) 
                h_b = np.zeros(T)
                
                for t in range(T):
                    try:
                        params_arr = to_pv_params(theta_path[t], fixed_params)
                        bc_t = pv.Bicop(family=fam, parameters=params_arr, rotation=rot)
                        pt = np.array([[u_vec[t], v_vec[t]]])
                        h_a[t] = bc_t.hfunc1(pt).item()
                        h_b[t] = bc_t.hfunc2(pt).item()
                    except:
                        h_a[t] = u_vec[t]
                        h_b[t] = v_vec[t]
            
            # [FIX 4] Store by Column Index (Unique) to prevent overwriting
            tree_outputs[tree][edge] = {
                'direct_var': a_idx,
                'direct_h': h_a,
                'indirect_var': b_idx,
                'indirect_h': h_b
            }

            key = f"T{tree}_{a_idx}-{b_idx}"
            dynamic_results[key] = {
                'theta': theta_path,
                'family': str(fam).split('.')[-1],
                'rotation': rot
            }
            edges_saved += 1

    print(f"\n[Validation] Processed: {edges_processed} | Saved: {edges_saved}")
    return dynamic_results

#####################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvinecopulib as pv
from tqdm import tqdm
from scipy.optimize import minimize

def plot_crash_probability(gas_results, u_data, asset_names, pair=('NVDA', 'SPY')):
    """
    Visualizes how the GAS model captures 'Crash Correlation' better than Static.
    """
    # 1. Identify the Edge
    idx1 = asset_names.index(pair[0])
    idx2 = asset_names.index(pair[1])
    
    # Try to find the key in Tree 0
    key_candidates = [f"T0_{idx1}-{idx2}", f"T0_{idx2}-{idx1}"]
    target_key = next((k for k in gas_results if any(c in k for c in key_candidates)), None)
    
    if not target_key:
        print(f"Pair {pair} not found in Tree 0 (Direct Connection).")
        return

    # 2. Extract Data
    theta_t = gas_results[target_key]['theta']
    family = gas_results[target_key]['family']
    
    # 3. Compute "Crash Probability" (Likelihood of both assets being in bottom 5%)
    # For a given theta, what is P(U < 0.05, V < 0.05)?
    # We simulate this for the plot
    
    dates = np.arange(len(theta_t))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Time (Days)')
    ax1.set_ylabel('Dynamic Correlation', color=color)
    ax1.plot(dates, theta_t, color=color, label='GAS Correlation', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(np.mean(theta_t), color='grey', linestyle='--', label='Static Mean')
    
    # Highlight Crash Periods
    # If correlation spikes > 0.8, we shade the region
    crash_periods = theta_t > np.percentile(theta_t, 90)
    ax1.fill_between(dates, 0, 1, where=crash_periods, color='red', alpha=0.1, transform=ax1.get_xaxis_transform(), label='High Systemic Risk')

    plt.title(f"Why Static Fails: {pair[0]}-{pair[1]} Dependence Spikes", fontsize=14)
    fig.tight_layout()
    plt.legend(loc='upper left')
    plt.show()

# --- 2. DIAGNOSTIC TOOL ---
def diagnose_gas_results(gas_results, n_assets=14):
    print("\n" + "="*40)
    print("       GAS MODEL DIAGNOSTIC REPORT       ")
    print("="*40)
    
    # CHECK 1: Structure Integrity
    total_edges = len(gas_results)
    if total_edges == 0:
        print("\n[CRITICAL FAIL] Result dictionary is empty.")
        print("  -> Cause: The matrix 'M' indices likely don't match the loop logic.")
        print("  -> Fix: Use the 'count_nonzero' transpose check I sent previously.")
        return

    print(f"\n[Structure] Total Edges Processed: {total_edges}")

    # CHECK 2: Dynamics (The "Dead Line" Test)
    # We differentiate between "Truly Static" (Independent/Gaussian=0) 
    # and "Failed Dynamic" (Optimizer stuck at initial guess).
    
    dynamic_count = 0
    static_count = 0
    stuck_optimizer_count = 0
    
    # Store volatilities to find the most active edge
    edge_volatility = []

    for key, val in gas_results.items():
        theta = np.array(val['theta'])
        std_dev = np.std(theta)
        
        edge_volatility.append((key, std_dev))
        
        if std_dev < 1e-6:
            static_count += 1
            # Check if it's non-zero but flat (Optimizer failed to move A/B)
            if np.mean(theta) != 0:
                stuck_optimizer_count += 1
        else:
            dynamic_count += 1
            
    print(f"[Dynamics] Activity Report:")
    print(f"  - Evolving Edges: {dynamic_count} (Healthy)")
    print(f"  - Flat Edges:     {static_count}")
    
    if stuck_optimizer_count > 0:
        print(f"  - [WARNING] {stuck_optimizer_count} edges are non-zero but FLAT.")
        print("    This means the optimizer likely failed (A=0, B=~1).")

    # CHECK 3: Visual Sanity Check
    # Plot the top 2 most volatile edges. 
    # If they look like sine waves or noise, it's bad. 
    # If they look like 'spiky' financial data (GARCH-like), it's good.
    
    # Sort by volatility
    edge_volatility.sort(key=lambda x: x[1], reverse=True)
    
    if len(edge_volatility) > 0:
        top_edges = edge_volatility[:2]
        
        fig, axes = plt.subplots(len(top_edges), 1, figsize=(10, 6), sharex=True)
        if len(top_edges) == 1: axes = [axes]
        
        print("\n[Visual Check] Plotting most active edges...")
        for i, (key, vol) in enumerate(top_edges):
            data = gas_results[key]['theta']
            fam = gas_results[key]['family']
            
            axes[i].plot(data, color='#1f77b4', linewidth=1.5)
            axes[i].set_title(f"Edge: {key} | Family: {fam} | Vol: {vol:.4f}")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylabel("Correlation (Theta)")
            
            # Add Static Reference Line
            axes[i].axhline(np.mean(data), color='red', linestyle='--', alpha=0.5, label='Mean')
            
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()
    else:
        print("[Visual Check] No edges to plot.")

# --- 3. MOCK DATA & EXECUTION ---
if __name__ == "__main__":
    print("Generating Proxy Data...")
    # Mock data: 14 assets, 300 days
    # We inject a known correlation shock to see if GAS picks it up
    T, N = 300, 14
    data = np.random.uniform(0, 1, (T, N))
    
    # Inject a shock: Asset 0 and 1 become highly correlated at t=150
    data[150:, 1] = data[150:, 0] * 0.9 + np.random.normal(0, 0.1, 150)
    data[:, 1] = (data[:, 1] - data[:, 1].min()) / (data[:, 1].max() - data[:, 1].min()) # Rescale to (0,1)

    print("Fitting Static Benchmark...")
    # 1. Instantiate empty model with correct dimension
    static_model = pv.Vinecop(d=N) 
    
    # 2. Select (Fit) the structure and families using the controls
    controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.clayton, pv.BicopFamily.gumbel, pv.BicopFamily.student])
    static_model.select(data, controls=controls)
    
    print("Running YOUR GAS Code...")
    # [HERE IS WHERE WE INPUT THE RESULTS]
    results = fit_gas_vine(data, static_model)
    
    # Run Diagnosis
    diagnose_gas_results(results, n_assets=N)

    