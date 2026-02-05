import networkx as nx
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pyvinecopulib as pv
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import stdtrit, stdtr
from matplotlib.lines import Line2D

# Exact Inverse Student-t CDF bridging Scipy (forward) and PyTorch (backward)
class InverseStudentT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, nu):
        u_cpu = u.detach().cpu().numpy()
        nu_cpu = nu.detach().cpu().numpy()
        # Clamp to avoid inf at boundaries
        u_cpu = np.clip(u_cpu, 1e-9, 1 - 1e-9)
        x = stdtrit(nu_cpu, u_cpu)
        x_tensor = torch.from_numpy(x).to(u.device, dtype=u.dtype)
        ctx.save_for_backward(x_tensor, nu)
        return x_tensor

    @staticmethod
    def backward(ctx, grad_output):
        x, nu = ctx.saved_tensors
        pi = torch.tensor(3.1415926535, device=x.device)
        
        # Log-space PDF calculation for stability
        log_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * pi)
        log_kernel = -((nu + 1) / 2) * torch.log(1 + (x**2) / nu)
        log_pdf = log_const + log_kernel
        
        pdf = torch.exp(log_pdf)
        # Prevent division by zero if PDF underflows in deep tails
        pdf = torch.clamp(pdf, min=1e-12) 
        
        grad_u = grad_output / pdf 
        return grad_u, None

def inverse_t_cdf(u, nu):
    return InverseStudentT.apply(u, nu)


# Single pair-copula whose parameter evolves according to the GAS rule
class GASPairCopula(nn.Module):
    def __init__(self, family, rotation=0):
        super().__init__()
        self.family = str(family).split('.')[-1].lower()
        self.rotation = int(rotation)

        # GAS Parameters (omega, A, B)
        self.omega = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.tensor(0.05))
        self.B = nn.Parameter(torch.tensor(0.9))

        # Additional parameter for Student-t copula
        if 'student' in self.family:
            self.nu_unconstrained = nn.Parameter(torch.tensor(1.5)) 
        else:
            self.register_parameter('nu_unconstrained', None)

    # Returns valid degrees of freedom > 2.0
    def get_nu(self):
        if self.nu_unconstrained is None: return None
        return torch.clamp(torch.nn.functional.softplus(self.nu_unconstrained) + 2.01, 2.01, 30)

    # Handles data rotation for copula families
    def rotate_data(self, u, v):
        if self.rotation == 90: return 1-u, v
        if self.rotation == 180: return 1-u, 1-v
        if self.rotation == 270: return u, 1-v
        return u, v
    
    # Maps unbounded GAS factor f_t to valid copula parameter space
    def transform_parameter(self, f_t):
        if 'gaussian' in self.family or 'student' in self.family:
            return torch.tanh(f_t) 
        elif 'clayton' in self.family:
            return torch.clamp(torch.nn.functional.softplus(f_t) + 1e-4, 1e-4, 20.0)
        elif 'gumbel' in self.family or 'joe' in self.family:
            return torch.clamp(torch.nn.functional.softplus(f_t) + 1.0 + 1e-4, 1.001, 20.0)
        elif 'frank' in self.family:
            val = f_t + torch.sign(f_t) * 1e-4 if torch.abs(f_t) < 1e-4 else f_t
            return torch.clamp(val, -30.0, 30.0)
        return f_t

    # Compute the log-likelihood for a batch of (u, v) given theta
    def log_likelihood_pair(self, u, v, theta, nu=None):

        # Rotate and clamp
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-6
        u_rot = torch.clamp(u_rot, eps, 1 - eps)
        v_rot = torch.clamp(v_rot, eps, 1 - eps)

        if 'gaussian' in self.family:
            rho = torch.clamp(theta, -0.999, 0.999)
            n = torch.distributions.Normal(0, 1)
            x, y = n.icdf(u_rot), n.icdf(v_rot)

            # Gaussian Copula Log-PDF  
            rho2 = rho**2      
            z = x**2 + y**2 - 2*rho*x*y
            log_det = 0.5 * torch.log(1 - rho2 + 1e-8)
            log_exp = -0.5 * (z / (1 - rho2 + 1e-8) - (x**2 + y**2))

            return -log_det + log_exp

        elif 'student' in self.family:
            rho = torch.clamp(theta, -0.999, 0.999)
            if nu is None: nu = torch.tensor(5.0, device=u.device)
            
            # Transform Uniforms (u,v) -> T-Scores (x,y)
            x = inverse_t_cdf(u_rot, nu)
            y = inverse_t_cdf(v_rot, nu)
            
            # Joint Log-Density: log c(u,v) = log(f_mv(x,y)) - log(f(x)) - log(f(y))
            rho2 = rho**2
            log_det = -0.5 * torch.log(1 - rho2 + 1e-8)
            zeta = (x**2 + y**2 - 2*rho*x*y) / (1 - rho2 + 1e-8)

            joint_kernel = -((nu + 2) / 2) * torch.log(1 + zeta / nu)
            marg_kernel_x = ((nu + 1) / 2) * torch.log(1 + (x**2) / nu)
            marg_kernel_y = ((nu + 1) / 2) * torch.log(1 + (y**2) / nu)
            
            log_gamma_const = torch.lgamma((nu + 2) / 2) + torch.lgamma(nu / 2) - 2 * torch.lgamma((nu + 1) / 2)
            return log_gamma_const + log_det + joint_kernel + marg_kernel_x + marg_kernel_y

        elif 'clayton' in self.family:
            t = theta
            a = torch.log(1+t) - (1+t)*(torch.log(u_rot)+torch.log(v_rot))
            b = torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1
            b = torch.clamp(b, min=eps)
            c = -(2 + 1/t) * torch.log(b)
            return a + c

        elif 'gumbel' in self.family:
            t = theta
            x, y = -torch.log(u_rot), -torch.log(v_rot)
            x = torch.clamp(x, min=eps)
            y = torch.clamp(y, min=eps)
            sum_pow = torch.pow(x**t + y**t, 1/t)
            return torch.log(sum_pow + t - 1) - sum_pow + (t-1)*(torch.log(x)+torch.log(y)) - torch.log(u_rot*v_rot) - 2*torch.log(u_rot*v_rot) # Simplified correction

        elif 'frank' in self.family:
            t = theta
            et = torch.exp(-t); eu = torch.exp(-t*u_rot); ev = torch.exp(-t*v_rot)
            num = t * (1-et) * eu * ev
            den = (1-et) - (1-eu)*(1-ev)
            return torch.log(torch.abs(num)) - 2*torch.log(torch.abs(den))

        return torch.zeros_like(u)

    # Computes h(u|v) = dC/dv for vine propagation
    def compute_h_func(self, u, v, theta, nu=None):
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-5
        u_rot = torch.clamp(u_rot, eps, 1-eps)
        v_rot = torch.clamp(v_rot, eps, 1-eps)
        
        h_val = torch.zeros_like(u)
        if 'gaussian' in self.family:
            rho = torch.clamp(theta, -0.999, 0.999)
            n = torch.distributions.Normal(0,1)
            x = n.icdf(u_rot)
            y = n.icdf(v_rot)
            denom = torch.sqrt(1 - rho**2 + 1e-8) 

            h_val = n.cdf((x - rho*y) / denom)

        elif 'student' in self.family:
            if nu is None: nu = torch.tensor(5.0)
            rho = torch.clamp(theta, -0.999, 0.999)
            
            u_in = u_rot.detach()
            v_in = v_rot.detach()
            
            x = inverse_t_cdf(u_in, nu)
            y = inverse_t_cdf(v_in, nu)
            
            num = x - rho * y
            den = torch.sqrt((1 - rho**2) * (nu + y**2) / (nu + 1) + 1e-8)
            arg = (num / den).cpu().numpy()
            df_np = (nu + 1).detach().cpu().numpy()

            cdf_val = stdtr(df_np, arg)
            h_val = torch.tensor(cdf_val, dtype=u.dtype, device=u.device)

        elif 'clayton' in self.family:
            t = theta

            # Formula: v^(-t-1) * (u^-t + v^-t - 1)^(-1/t - 1)
            term1 = -(t + 1) * torch.log(v_rot)
            base = torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1
            base = torch.clamp(base, min=eps)
            term2 = -(1/t + 1) * torch.log(base)

            h_val = torch.exp(term1 + term2)

        elif 'gumbel' in self.family:
            t = theta
            x = -torch.log(u_rot)
            y = -torch.log(v_rot)
            x = torch.clamp(x, min=eps)
            y = torch.clamp(y, min=eps)

            # Formula: C(u,v) * [(-ln v)^(t-1) / v] * [(-ln u)^t + (-ln v)^t]^(1/t - 1)
            sum_pow = torch.pow(x**t + y**t, 1/t)
            term1 = sum_pow + (t-1)*torch.log(x)
            term2 = torch.log(x**t + y**t) * (1/t - 1)

            h_val = torch.exp(-sum_pow + (t-1)*torch.log(x) + (1/t - 1)*torch.log(x**t + y**t)) / u_rot

        elif 'frank' in self.family:
            t = theta
            eu = torch.exp(-t * u_rot) - 1
            num = eu * torch.exp(-t * v_rot)
            den = (1 - torch.exp(-t)) - (1 - torch.exp(-t * u_rot)) * (1 - torch.exp(-t * v_rot))
            den = torch.sign(den) * torch.max(torch.abs(den), torch.tensor(eps, device=u.device))
            
            h_val = num / den

        # Un-Rotate H-Function
        if self.rotation == 90: return 1 - h_val
        if self.rotation == 270: return 1 - h_val

        return torch.clamp(h_val, 1e-6, 1 - 1e-6)

    def forward(self, u_data, v_data):
        T = u_data.shape[0]
        f_t = self.omega / (1 - self.B)  # Initialize at unconditional mean
        nu_val = self.get_nu()

        log_likes = []
        thetas = []
        for t in range(T):
            theta_t = self.transform_parameter(f_t)
            thetas.append(theta_t)
            u_t = u_data[t:t+1]
            v_t = v_data[t:t+1]
            
            # Make f_t a leaf variable for Autograd
            f_t_leaf = f_t.detach().requires_grad_(True)
            theta_leaf = self.transform_parameter(f_t_leaf)
            ll = self.log_likelihood_pair(u_t, v_t, theta_leaf, nu_val)
            
            # Compute Score = dLL/df_t via Autograd
            score = torch.autograd.grad(ll, f_t_leaf, create_graph=True, allow_unused=True)[0]

            if score is None or torch.isnan(score) or torch.isinf(score):
                score = torch.tensor(0.0, device=u_data.device) # Neutralize the shock
            else:
                score = torch.clamp(score, -5.0, 5.0) # Prevent explosion
            
            # GAS Update
            f_next = self.omega + self.A * score + self.B * f_t
            if torch.isnan(f_next) or torch.isinf(f_next):
                f_next = f_t.detach() * 0.98

            log_likes.append(ll)
            f_t = f_next.detach()
            
        return -torch.sum(torch.stack(log_likes)), torch.stack(thetas)

# Fits GAS dynamics to a generic R-Vine structure using Dißmann algorithm
def fit_mixed_gas_vine(u_matrix, structure):
    T, N = u_matrix.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_tensor = torch.tensor(u_matrix, dtype=torch.float32).to(device)
    
    # Parse Structure Matrix
    M = np.array(structure.matrix)
    if M.max() == N: M = np.where(M > 0, M - 1, M)
        
    fams = structure.pair_copulas
    vine_data = torch.zeros((N, N, T), device=device)
    
    # Initialize Triangular Grid with rax data
    for i in range(N):
        var_idx = int(M[i, i])
        vine_data[i, i] = u_tensor[:, var_idx]

    fitted_paths = {}
    print(f"Fitting R-Vine GAS Model on {N} assets...")

    # Tree Iteration from the bottom tree (unconditional) up to the root
    for tree in range(N - 1):
        print(f"--- Processing Tree {tree + 1} / {N - 1} ---")
        edges = N - 1 - tree
        
        for edge in range(edges):
            # Dißmann Index Logic for R-Vine Matrix
            m_row = N - 1 - tree
            m_col = edge
            
            # Determine Inputs (u, v)
            if tree == 0:
                # Tree 0 (Bottom): Inputs are raw variables.
                # In R-Vine matrix, the pair at (m_row, m_col) connects:
                # 1. Variable at M[m_row, m_col]
                # 2. Variable at M[m_col, m_col] (Diagonal element of the column)
                var1_idx = int(M[m_row, m_col])
                var2_idx = int(M[m_col, m_col]) 
                
                u_vec = u_tensor[:, var1_idx]
                v_vec = u_tensor[:, var2_idx]

                # For storage later: we need to know where 'v' "lives" in the matrix row
                # In the bottom row, var2 is just the diagonal variable.
                # We identify it by finding which column p in this row holds var2_idx.
                partner_col = -1
                for p in range(N):
                    if M[m_row, p] == var2_idx:
                        partner_col = p
                        break
            else:
                # Tree > 0: Inputs are h-functions from previous tree (stored in grid).
                target_var = M[m_row, m_col]
                
                # Search for the "partner" column in the row below
                partner_col = -1
                for p in range(m_col + 1, N):
                    if M[m_row + 1, p] == target_var:
                        partner_col = p
                        break
                
                if partner_col == -1: partner_col = m_row + 1
                
                # Fetch transformed variables
                u_vec = vine_data[m_row + 1, m_col]
                v_vec = vine_data[m_row + 1, partner_col]

            # NAN FILLER FOR INPUTS
            if torch.isnan(u_vec).any(): u_vec = torch.nan_to_num(u_vec, 0.5)
            if torch.isnan(v_vec).any(): v_vec = torch.nan_to_num(v_vec, 0.5)

            # Fit GAS Model
            pc = fams[tree][edge]
            fam_str = str(pc.family)
            rot = pc.rotation
            
            # Independence Copula Check
            if 'indep' in fam_str.lower():
                fitted_paths[f"T{tree}_E{edge}"] = torch.zeros(T)
                h_direct = u_vec
                h_indirect = v_vec
            else:
                # Initialize and Train
                model = GASPairCopula(fam_str, rotation=rot).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.02)
                
                for _ in range(50):
                    optimizer.zero_grad()
                    loss, _ = model(u_vec.unsqueeze(1), v_vec.unsqueeze(1))
                    if torch.isnan(loss): break 
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Extract Path
                _, theta_path = model(u_vec.unsqueeze(1), v_vec.unsqueeze(1))
                theta_path = torch.nan_to_num(theta_path, 0.0) # Safety
                fitted_paths[f"T{tree}_E{edge}"] = theta_path.detach().cpu().numpy()

                # Compute H-functions for next level
                nu_val = model.get_nu()
                h_d_list = []
                h_i_list = []

                with torch.no_grad():
                    h_direct = model.compute_h_func(u_vec, v_vec, theta_path, nu_val)
                    h_indirect = model.compute_h_func(v_vec, u_vec, theta_path, nu_val)

            # Update Grid
            vine_data[m_row, m_col] = h_direct
            if partner_col != -1 and partner_col != m_col:
                 vine_data[m_row, partner_col] = h_indirect
            elif tree == 0 and partner_col != -1:
                 vine_data[m_row, partner_col] = h_indirect
            
    return fitted_paths, structure

# Visualizes the R-Vine structure (Matrix + Families)
def plot_vine_structure(structure, num_assets):
    """
    1. Plots the Network Graph of Tree 1 (The most important dependencies).
    2. Prints a summary distribution of Copula Families across all trees.
    """
    M = np.array(structure.matrix)
    if M.max() == num_assets: M = np.where(M > 0, M - 1, M) # 0-based fix
    
    fams = structure.pair_copulas
    
    # --- 1. VISUALIZE TREE 1 (Network Graph) ---
    print("\n--- Generating Tree 1 Network Graph ---")
    G = nx.Graph()
    
    # Add nodes (Assets 0 to N-1)
    for i in range(num_assets):
        G.add_node(i, label=f"Asset {i}")
    
    # Edges in Tree 1 are defined by the bottom row of the R-Vine Matrix
    # Pair at col j connects: M[N-1, j] <--> M[j, j]
    row = num_assets - 1
    edge_colors = []
    families_tree1 = []
    
    for j in range(num_assets - 1):
        u = M[row, j] # First variable
        v = M[j, j]   # Second variable (diagonal)
        
        # Get Family
        pc = fams[0][j]
        fam_name = str(pc.family).split('.')[-1]
        families_tree1.append(fam_name)
        
        G.add_edge(u, v, family=fam_name)
        
        # Color coding
        if 'clayton' in fam_name.lower(): edge_colors.append('red')      # Crash Risk
        elif 'gumbel' in fam_name.lower(): edge_colors.append('green')   # Boom Risk
        elif 'student' in fam_name.lower(): edge_colors.append('purple') # Fat Tails
        elif 'gaussian' in fam_name.lower(): edge_colors.append('blue')  # Standard
        elif 'frank' in fam_name.lower(): edge_colors.append('orange')   # Symmetric
        else: edge_colors.append('gray')

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42) # k regulates distance
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightgray')
    nx.draw_networkx_labels(G, pos, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors)
    
    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Clayton (Lower Tail)'),
        Line2D([0], [0], color='green', lw=2, label='Gumbel (Upper Tail)'),
        Line2D([0], [0], color='purple', lw=2, label='Student-t (Fat Tails)'),
        Line2D([0], [0], color='blue', lw=2, label='Gaussian (Linear)'),
        Line2D([0], [0], color='orange', lw=2, label='Frank (Symmetric)'),
        Line2D([0], [0], color='gray', lw=2, label='Independence')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.title("Tree 1 Topology: Dominant Market Dependencies")
    plt.axis('off')
    plt.show()

    # --- 2. SUMMARY STATISTICS (All Trees) ---
    print("\n--- Copula Family Distribution ---")
    all_families = []
    for tree in fams:
        for pc in tree:
            all_families.append(str(pc.family).split('.')[-1])
            
    df_fam = pd.DataFrame(all_families, columns=['Family'])
    counts = df_fam['Family'].value_counts(normalize=True) * 100
    
    print(counts)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.ylabel("Percentage (%)")
    plt.title("Distribution of Copula Families across all Trees")
    plt.show()

if __name__ == "__main__":
    print("1. Generating Synthetic 14-Asset Data...")
    T = 1250 
    N = 14
    # Create random covariance
    A = np.random.rand(N, N)
    cov = np.dot(A, A.transpose())
    data_mvn = np.random.multivariate_normal(np.zeros(N), cov, size=T)
    data_u = norm.cdf(data_mvn)
    
    print("2. Selecting Structure via pyvinecopulib...")
    controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.clayton, pv.BicopFamily.gumbel, pv.BicopFamily.student, pv.BicopFamily.frank])
    structure = pv.Vinecop(d=N) 
    structure.select(data_u, controls=controls)
    print(f"   Structure: R-Vine with {len(structure.pair_copulas)} trees.")

    print("3. Fitting Dynamic GAS Vine (Model 0)...")
    paths = fit_mixed_gas_vine(data_u, structure)
    
    print("4. Plotting Results...")
    keys = list(paths.keys())
    plt.figure(figsize=(12, 6))
    # Plot first edge of first tree
    plt.plot(paths[keys[0]], label=f"Tree 1 Edge 1 ({keys[0]})")
    # Plot last edge
    plt.plot(paths[keys[-1]], label=f"Deep Edge ({keys[-1]})")
    plt.title("Recovered Dynamic Dependence (GAS Parameters)")
    plt.legend()
    plt.show()

    plot_vine_structure(structure, N)
    
    print("Done")