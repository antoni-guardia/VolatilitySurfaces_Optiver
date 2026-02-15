import numpy as np
import pandas as pd
from arch import arch_model
from EVT import EVT
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import probplot, kstest, norm, uniform
import os

class GJR_GARCH_EVT:
    def __init__(self):
        self.garch_model = None
        self.evt_model = None
        self.params = {}
        
        # Data storage for diagnostics
        self.train_uniforms = None
        self.test_uniforms = None
        self.resids = None 
        self.test_resids = None
        self.vol = None
        self.test_vol = None
        
    def fit_and_predict(self, series, holdout_days=252):
        s = pd.Series(series)
        r_full = s.values
        idx_full = s.index
        
        # 1. Strict Train/Test Split
        r_train, idx_train = r_full[:-holdout_days], idx_full[:-holdout_days]
        r_test, idx_test = r_full[-holdout_days:], idx_full[-holdout_days:]
        
        # 2. Fit GJR-GARCH on TRAIN ONLY
        # p=1 (GARCH), o=1 (Asymmetric/Leverage term -> GJR), q=1 (ARCH)
        garch = arch_model(r_train, vol='GARCH', p=1, o=1, q=1, mean='Constant', dist='normal', rescale=False)
        garch_res = garch.fit(disp='off', show_warning=False)
        
        self.garch_model = garch_res
        self.params = {
            'mu': garch_res.params['mu'],
            'omega': garch_res.params['omega'], 
            'alpha': garch_res.params['alpha[1]'],
            'gamma': garch_res.params['gamma[1]'], # This is the GJR asymmetric leverage term!
            'beta': garch_res.params['beta[1]'], 
            'loglikelihood': garch_res.loglikelihood
        }
        
        # 3. Filter FULL series using TRAIN params (No Data Leakage)
        mu = self.params['mu']
        omega = self.params['omega']
        alpha = self.params['alpha']
        gamma = self.params['gamma']
        beta = self.params['beta']
        
        vol_full = np.zeros(len(r_full))
        vol_full[0] = np.var(r_train)
        
        # GJR-GARCH Volatility Filter
        for t in range(1, len(r_full)):
            eps_prev = r_full[t-1] - mu
            # Indicator function: 1 if previous shock was negative, 0 otherwise
            I_prev = 1.0 if eps_prev < 0 else 0.0 
            vol_full[t] = omega + (alpha + gamma * I_prev) * (eps_prev**2) + beta * vol_full[t-1]
            
        vol_full = np.sqrt(vol_full)
        z_full = (r_full - mu) / vol_full
        
        z_train = z_full[:-holdout_days]
        z_test = z_full[-holdout_days:]
        
        # 4. Fit EVT on TRAIN ONLY, transform both
        evt = EVT()
        # Equity tails are fat, 10% is a great threshold
        evt.fit(z_train, lower_quantile=0.10, upper_quantile=0.10)
        self.evt_model = evt
                
        self.train_uniforms = pd.Series(evt.transform(z_train), index=idx_train)
        self.test_uniforms = pd.Series(evt.transform(z_test), index=idx_test)
        
        # Save states for the diagnostic plotter
        self.resids = z_train
        self.test_resids = z_test
        self.vol = vol_full[:-holdout_days]
        self.test_vol = vol_full[-holdout_days:]
        
        return self
    
    def diagnostics(self, name="Asset", is_test=False, save=None):
        # Swap between Train and Test data
        mu = self.params['mu']
        z = self.test_resids if is_test else self.resids
        u_series = self.test_uniforms if is_test else self.train_uniforms
        u = u_series.values
        vol_series = self.test_vol if is_test else self.vol
        
        phase = "Out-of-Sample (Test)" if is_test else "In-Sample (Train)"
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        fig.suptitle(f'GJR-GARCH-EVT Diagnostics [{phase}]: {name}', fontsize=16, fontweight='bold')
        
        # 1. Realized Returns vs Predicted Volatility Bands (VaR)
        ax0 = fig.add_subplot(gs[0, :])
        actual_returns = z * vol_series + mu
        
        # For GARCH-EVT, the 95% band is derived from the EVT distribution, not Student-t!
        # An easy visual approximation is +/- 1.96 standard deviations, but since we have heavy tails,
        # we can just use the empirical/EVT 95% quantile from the training residuals:
        q975 = np.percentile(self.resids, 97.5) 
        q025 = np.percentile(self.resids, 2.5)
        
        ax0.plot(u_series.index, actual_returns, lw=1, alpha=0.6, color='black', label='Realized Returns')
        ax0.plot(u_series.index, mu + q975 * vol_series, 'r--', lw=1.5, label='Upper 97.5% Volatility Band')
        ax0.plot(u_series.index, mu + q025 * vol_series, 'g--', lw=1.5, label='Lower 2.5% Volatility Band (VaR)')
        ax0.set_title('Realized Returns vs. Predicted Conditional Volatility')
        ax0.legend(loc='upper left')
        ax0.grid(alpha=0.3)
        
        # 2. Histogram with EVT fit
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.hist(z, bins=50, density=True, alpha=0.6, color='steelblue', ec='black')
        x_range = np.linspace(z.min(), z.max(), 200)
        ax1.plot(x_range, norm.pdf(x_range), 'r--', lw=2, label='N(0,1)')
        ax1.axvline(self.evt_model.u_lower, color='orange', ls='--', lw=1.5, label='EVT thresholds')
        ax1.axvline(self.evt_model.u_upper, color='orange', ls='--', lw=1.5)
        ax1.set_title('Distribution with EVT Thresholds')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 3. QQ plot (Normal core)
        ax2 = fig.add_subplot(gs[1, 1])
        probplot(z, dist=norm, plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Core)')
        ax2.grid(alpha=0.3)
        
        # 4. ACF residuals
        ax3 = fig.add_subplot(gs[1, 2])
        plot_acf(z, lags=40, ax=ax3, alpha=0.05)
        ax3.set_title('ACF Residuals')
        ax3.grid(alpha=0.3)
        
        # 5. ACF squared residuals
        ax4 = fig.add_subplot(gs[2, 0])
        plot_acf(z**2, lags=40, ax=ax4, alpha=0.05)
        ax4.set_title('ACF Squared Residuals')
        ax4.grid(alpha=0.3)
        
        # 6. Uniform histogram (PIT)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(u, bins=50, density=True, alpha=0.6, color='purple', ec='black')
        ax5.axhline(1.0, color='r', ls='--', lw=2, label='Uniform')
        ks_stat, ks_p = kstest(u, 'uniform')
        ax5.text(0.05, 0.95, f'KS: {ks_stat:.3f}\np={ks_p:.3f}',
                transform=ax5.transAxes, va='top',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
        ax5.set_title('Uniform Transform (EVT)')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 7. QQ Uniform
        ax6 = fig.add_subplot(gs[2, 2])
        probplot(u, dist=uniform, plot=ax6)
        ax6.set_title('Q-Q Uniform')
        ax6.grid(alpha=0.3)
        
        plt.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.93, hspace=0.5, wspace=0.3)
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        return fig


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    data_path = os.path.join(project_root, "data", "processed", "returns.csv")
    res_dir = os.path.join(project_root, "outputs", "dynamics")
    diag_dir = os.path.join(res_dir, "plots", "gjr_garch_evt")
    os.makedirs(diag_dir, exist_ok=True)
    
    HOLDOUT_DAYS = 252 # 1 Year Split
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).apply(pd.to_numeric, errors='coerce')
    print(f"Shape: {df.shape}, Assets: {df.columns.tolist()}\n")
    
    train_u, test_u, params = {}, {}, []
    
    for col in df.columns:        
        print(f"Fitting GJR-GARCH-EVT for: {col}")
        m = GJR_GARCH_EVT()
        m.fit_and_predict(df[col].dropna(), holdout_days=HOLDOUT_DAYS)
        
        train_u[col] = m.train_uniforms
        test_u[col] = m.test_uniforms
        
        # GENERATE BOTH GRAPHS
        m.diagnostics(name=col, is_test=False, save=os.path.join(diag_dir, f"{col}_train_diag.png"))
        m.diagnostics(name=col, is_test=True,  save=os.path.join(diag_dir, f"{col}_test_diag.png"))
        plt.close('all')

        p = m.params.copy()
        p['asset'] = col
        p['persistence'] = p['alpha'] + p['gamma']/2 + p['beta'] # GJR persistence formula
        params.append(p)
    
    # ENFORCE DATA CONTRACT AND SAVE
    train_df, test_df = pd.DataFrame(train_u), pd.DataFrame(test_u)
    train_df.index, test_df.index = pd.to_datetime(train_df.index).date, pd.to_datetime(test_df.index).date
    train_df.index.name, test_df.index.name = "Date", "Date"
    
    # Save as ngarch_t simply to avoid having to rename files in the copula scripts, 
    # but strictly speaking this is GJR-GARCH-EVT now!
    train_df.to_csv(os.path.join(res_dir, "train_uniforms_gjr_garch.csv"))
    test_df.to_csv(os.path.join(res_dir, "test_uniforms_gjr_garch.csv"))

    # ENFORCE DATA CONTRACT ON PARAMS
    p_df = pd.DataFrame(params)
    p_df = p_df[['asset'] + [c for c in p_df.columns if c != 'asset']]
    p_df.to_csv(os.path.join(res_dir, "gjr_garch_evt_params.csv"), index=False)
