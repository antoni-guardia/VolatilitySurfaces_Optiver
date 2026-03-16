import numpy as np
import torch
import pyvinecopulib as pv
from scipy.stats import norm, t as student_t

# Import your PyTorch Copula modules
from src.dependence.gas_mixed_vine import GASPairCopula
from src.dependence.neural_mixed_vine import NeuralPairCopula

class DynamicGASVine:
    def __init__(self, pth_path, static_json_path):
        self.base_copula = pv.Vinecop.from_file(static_json_path)
        self.matrix = np.array(self.base_copula.matrix, dtype=np.int64)
        self.N = self.matrix.shape[0]
        self.models = {}
        
        gas_dict = torch.load(pth_path, map_location='cpu', weights_only=False)
        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                edge_key = f"T{tree}_E{edge}"
                info = gas_dict.get(edge_key, None)
                if not info or info['family'] == 'indep':
                    self.models[edge_key] = None
                    continue
                    
                model = GASPairCopula(info['family'], info['rotation'])
                model.omega.data = torch.tensor(info['omega'])
                model.A.data = torch.tensor(info['A'])
                b_val = info['B']
                model.B_logit.data = torch.tensor(np.log(b_val / (1 - b_val + 1e-8)) if b_val < 1.0 else 10.0)
                if model.nu_param is not None and not np.isnan(info.get('nu', np.nan)):
                    model.nu_param.data = torch.tensor(np.log(np.exp(max(info['nu'] - 2.01, 1e-6)) - 1))
                
                oos = float(info['oos_forecast'])
                if 'gaussian' in info['family'] or 'student' in info['family']: f_init = np.arctanh(np.clip(oos / 0.999, -0.99, 0.99))
                elif 'clayton' in info['family']: f_init = np.log(np.exp(max(oos - 1e-5, 1e-6)) - 1)
                elif 'gumbel' in info['family']: f_init = np.log(np.exp(max(oos - 1.0001, 1e-6)) - 1)
                else: f_init = oos
                    
                model.f_t = torch.tensor(f_init)
                self.models[edge_key] = model
        self._push_to_cpp()
        
    def _push_to_cpp(self):
        pcs_list = []
        trunc_lvl = len(self.base_copula.pair_copulas)
        for tree in range(trunc_lvl):
            tree_list = []
            for edge in range(self.N - 1 - tree):
                pc = self.base_copula.get_pair_copula(tree, edge)
                model = self.models[f"T{tree}_E{edge}"]
                if model:
                    # FIX: GAS natively stores its forecast in model.f_t
                    theta = float(model.transform_parameter(model.f_t).item())
                    nu = model.get_nu()
                    
                    if pc.parameters.shape[0] == 2:
                        nu_val = float(nu.item()) if nu is not None else 5.0
                        params = np.array([[np.clip(theta, -0.999, 0.999)], [nu_val]])
                    else:
                        if any(f in model.family for f in ['gaussian', 'student']): theta = np.clip(theta, -0.999, 0.999)
                        elif 'frank' in model.family: theta = np.clip(theta, -40.0, 40.0); theta = 1e-4 if abs(theta) < 1e-4 else theta
                        elif 'clayton' in model.family: theta = np.clip(theta, 1e-5, 28.0)
                        elif 'gumbel' in model.family: theta = np.clip(theta, 1.001, 28.0)
                        params = np.array([[theta]])
                    pc.parameters = params
                tree_list.append(pc)
            pcs_list.append(tree_list)
        self.base_copula = pv.Vinecop.from_structure(structure=self.base_copula.structure, pair_copulas=pcs_list)
        
    def simulate(self, n_scenarios): return np.clip(self.base_copula.simulate(n_scenarios), 1e-6, 1 - 1e-6)

    def update_states(self, u_realized_np):
        u_tensor = torch.tensor(u_realized_np, dtype=torch.float64).view(1, -1)
        M = self.matrix
        if M.max() == self.N: M -= 1 
        if np.sum(M[0] >= 0) > np.sum(M[-1] >= 0): M = np.flipud(M)
            
        h_storage = {(i, -1): u_tensor[:, i] for i in range(self.N)}
        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                row, col = self.N - 1 - tree, edge
                u_vec = h_storage[(M[row, col], -1)] if tree == 0 else h_storage[(col, tree-1)]
                var_2, partner_col = M[col, col], -1
                
                if tree == 0: v_vec = h_storage[(var_2, -1)]
                else:
                    for k in range(self.N):
                        if M[row+1, k] == var_2: partner_col = k; break
                    v_vec = h_storage[(partner_col, tree-1)]
                    
                model = self.models[f"T{tree}_E{edge}"]
                if model is None:
                    h_storage[(edge, tree)] = u_vec
                    if tree < self.N - 2: h_storage[(partner_col, tree)] = v_vec
                else:
                    _, _, h_dir, h_indir = model.update_step(u_vec, v_vec)
                    h_storage[(edge, tree)] = h_dir
                    if tree < self.N - 2: h_storage[(partner_col, tree)] = h_indir
        self._push_to_cpp()

class DynamicNeuralVine:
    def __init__(self, pth_path, static_json_path, history_window):
        self.base_copula = pv.Vinecop.from_file(static_json_path)
        self.matrix = np.array(self.base_copula.matrix, dtype=np.int64)
        self.N = self.matrix.shape[0]
        self.models = {}
        
        neural_dict = torch.load(pth_path, map_location='cpu', weights_only=False)

        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                edge_key = f"T{tree}_E{edge}"
                info = neural_dict.get(edge_key, None)

                if not info or info.get('family') == 'indep':
                    self.models[edge_key] = None
                    continue
                
                model = NeuralPairCopula(
                    family=info['family'], 
                    rotation=info.get('rotation', 0),
                    hidden_dim=info.get('hidden_dim', 8),
                    num_layers=info.get('num_layers', 1),
                    dropout=info.get('dropout', 0.0)
                )
                
                if 'state_dict' in info:
                    model.load_state_dict(info['state_dict'], strict=True)
                
                model.eval()
                self.models[edge_key] = model
                
        self._warm_up(history_window)
        self._push_to_cpp()

    def _warm_up(self, history_window):
        for t in range(history_window.shape[0]):
            self.update_states(history_window[t:t+1])
            
    def _push_to_cpp(self):
        pcs_list = []
        trunc_lvl = len(self.base_copula.pair_copulas)
        for tree in range(trunc_lvl):
            tree_list = []
            for edge in range(self.N - 1 - tree):
                pc = self.base_copula.get_pair_copula(tree, edge)
                model = self.models[f"T{tree}_E{edge}"]
                if model:
                    with torch.no_grad():
                        dummy_in = torch.zeros(1, 1, 2)
                        out, _ = model.rnn(dummy_in, model.hidden_state)
                        f_t = model.head(out).squeeze()
                        theta = float(model.transform_parameter(f_t).item())
                        nu = model.get_nu()
                    
                    if pc.parameters.shape[0] == 2:
                        nu_val = float(nu.item()) if nu is not None else 5.0
                        params = np.array([[np.clip(theta, -0.999, 0.999)], [nu_val]])
                    else:
                        if any(f in model.family for f in ['gaussian', 'student']): theta = np.clip(theta, -0.999, 0.999)
                        elif 'frank' in model.family: theta = np.clip(theta, -40.0, 40.0); theta = 1e-4 if abs(theta) < 1e-4 else theta
                        elif 'clayton' in model.family: theta = np.clip(theta, 1e-5, 28.0)
                        elif 'gumbel' in model.family: theta = np.clip(theta, 1.001, 28.0)
                        params = np.array([[theta]])
                    pc.parameters = params
                tree_list.append(pc)
            pcs_list.append(tree_list)
        self.base_copula = pv.Vinecop.from_structure(structure=self.base_copula.structure, pair_copulas=pcs_list)
                
    def simulate(self, n_scenarios): return np.clip(self.base_copula.simulate(n_scenarios), 1e-6, 1 - 1e-6)

    def update_states(self, u_realized_np):
        u_tensor = torch.tensor(u_realized_np, dtype=torch.float64).view(1, -1)
        M = self.matrix
        if M.max() == self.N: M -= 1 
        if np.sum(M[0] >= 0) > np.sum(M[-1] >= 0): M = np.flipud(M)
            
        h_storage = {(i, -1): u_tensor[:, i] for i in range(self.N)}
        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                row, col = self.N - 1 - tree, edge
                u_vec = h_storage[(M[row, col], -1)] if tree == 0 else h_storage[(col, tree-1)]
                var_2, partner_col = M[col, col], -1
                
                if tree == 0: v_vec = h_storage[(var_2, -1)]
                else:
                    for k in range(self.N):
                        if M[row+1, k] == var_2: partner_col = k; break
                    v_vec = h_storage[(partner_col, tree-1)]
                    
                model = self.models[f"T{tree}_E{edge}"]
                if model is None:
                    h_storage[(edge, tree)] = u_vec
                    if tree < self.N - 2: h_storage[(partner_col, tree)] = v_vec
                else:
                    _, _, h_dir, h_indir = model.step_forward(u_vec, v_vec)
                    h_storage[(edge, tree)] = h_dir
                    if tree < self.N - 2: h_storage[(partner_col, tree)] = h_indir
        self._push_to_cpp()

class UniversalScenarioGenerator:
    def __init__(self, factor_order, copula_model, model_id):
        self.factor_order = factor_order
        self.copula = copula_model 
        self.model_id = model_id
        self._ng_idx, self._har_idx, self._nsde_idx = [], [], []

    def classify_marginals(self, marginals):
        for i, n in enumerate(self.factor_order):
            m = marginals[n]
            if hasattr(m, 'pi_drift') and hasattr(m, 'pi_diff'): self._nsde_idx.append(i)
            elif hasattr(m, 'params') and isinstance(m.params, list): self._ng_idx.append(i)
            else: self._har_idx.append(i)

    def simulate_1day_dual(self, n_scenarios, init_states, marginals):
        if not self._ng_idx and not self._har_idx and not self._nsde_idx: 
            self.classify_marginals(marginals)
            
        dim = len(self.factor_order)
        paths_j = np.zeros((n_scenarios, dim))
        paths_i = np.zeros((n_scenarios, dim))
        
        U_j = np.clip(self.copula.simulate(n_scenarios), 1e-6, 1 - 1e-6)
        U_i = np.random.uniform(1e-6, 1 - 1e-6, size=(n_scenarios, dim))

        for d_idx in self._ng_idx:
            n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
            mu, omega, alpha, beta, theta, nu = m.params
            prev_sig = max(np.sqrt(init_states[n_name]['sigma2']), 1e-6)
            prev_z = init_states[n_name]['resid'] / prev_sig
            next_sig2 = omega + alpha * ((prev_z - theta)**2) * init_states[n_name]['sigma2'] + beta * init_states[n_name]['sigma2']
            sig_term = np.sqrt(next_sig2)
            paths_j[:, d_idx] = mu + sig_term * student_t.ppf(U_j[:, d_idx], df=nu)
            paths_i[:, d_idx] = mu + sig_term * student_t.ppf(U_i[:, d_idx], df=nu)

        for d_idx in self._har_idx:
            n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
            p, h = m.params, init_states[n_name]['history']
            mean = p['har_intercept'] + p['har_daily']*h[-1] + p['har_weekly']*h[-5:].mean() + p['har_monthly']*h.mean()
            sig_term = np.sqrt(p['garch_omega'] + p['garch_alpha']*(init_states[n_name]['resid']**2) + p['garch_beta']*init_states[n_name]['sigma2'])
            if hasattr(m, 'evt_model') and m.evt_model is not None:
                paths_j[:, d_idx] = mean + sig_term * m.evt_model.inverse_transform(U_j[:, d_idx])
                paths_i[:, d_idx] = mean + sig_term * m.evt_model.inverse_transform(U_i[:, d_idx])
            else:
                paths_j[:, d_idx] = mean + sig_term * norm.ppf(U_j[:, d_idx])
                paths_i[:, d_idx] = mean + sig_term * norm.ppf(U_i[:, d_idx])

        if self._nsde_idx:
            dt = 1.0 / 252.0
            for d_idx in self._nsde_idx:
                n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
                window = torch.tensor(init_states[n_name]['history'], dtype=torch.float64, device=m.device).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    mu_pred, sig_pred, nu = m.pi_drift(window).item(), m.pi_diff(window).item(), m.nu.item()
                paths_j[:, d_idx] = mu_pred * dt + sig_pred * np.sqrt(dt) * student_t.ppf(U_j[:, d_idx], df=nu)
                paths_i[:, d_idx] = mu_pred * dt + sig_pred * np.sqrt(dt) * student_t.ppf(U_i[:, d_idx], df=nu)

        return paths_j, paths_i

    def calculate_realized_uniforms(self, realized_row, init_states, marginals):
        u_realized = np.zeros(len(self.factor_order))
        for d_idx in self._ng_idx:
            n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
            mu, omega, alpha, beta, theta, nu = m.params
            prev_sig = max(np.sqrt(init_states[n_name]['sigma2']), 1e-6)
            next_sig2 = omega + alpha * ((init_states[n_name]['resid'] / prev_sig - theta)**2) * init_states[n_name]['sigma2'] + beta * init_states[n_name]['sigma2']
            u_realized[d_idx] = student_t.cdf((realized_row[d_idx] - mu) / np.sqrt(next_sig2), df=nu)

        for d_idx in self._har_idx:
            n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
            p, h = m.params, init_states[n_name]['history']
            mean = p['har_intercept'] + p['har_daily']*h[-1] + p['har_weekly']*h[-5:].mean() + p['har_monthly']*h.mean()
            sig2 = p['garch_omega'] + p['garch_alpha']*(init_states[n_name]['resid']**2) + p['garch_beta']*init_states[n_name]['sigma2']
            z_real = (realized_row[d_idx] - mean) / np.sqrt(sig2)
            u_realized[d_idx] = m.evt_model.transform(np.array([z_real]))[0] if hasattr(m, 'evt_model') else norm.cdf(z_real)

        if self._nsde_idx:
            dt = 1.0 / 252.0
            for d_idx in self._nsde_idx:
                n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
                window = torch.tensor(init_states[n_name]['history'], dtype=torch.float64, device=m.device).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    mu_pred, sig_pred, nu = m.pi_drift(window).item(), m.pi_diff(window).item(), m.nu.item()
                u_realized[d_idx] = student_t.cdf((realized_row[d_idx] - mu_pred * dt) / (sig_pred * np.sqrt(dt)), df=nu)
            
        return np.clip(u_realized, 1e-6, 1 - 1e-6)