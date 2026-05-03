"""
Generate the News Impact Curve (NIC) for SPY, OIL, and GOLD.
Visualizes why asymmetric models (like GJR-GARCH) or Machine Learning 
(with negative return features) outperform standard symmetric GARCH.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eda_outputs"
OUT_DIR.mkdir(exist_ok=True)

TARGETS = ["SPY", "OIL", "GOLD"]

def plot_news_impact_curve():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    shocks = np.linspace(-5, 5, 200)
    
    for idx, target in enumerate(TARGETS):
        print(f"Loading {target} daily training data...")
        DATA_PATH = ROOT / "data" / "splits" / "daily" / "no_exog" / target / "train.csv"
        
        if not DATA_PATH.exists():
            print(f"  Data for {target} not found. Skipping.")
            continue
            
        df = pd.read_csv(DATA_PATH)
        returns = df['ret'] * 100.0  # Convert to percentage
        ax = axes[idx]

        # 1. Standard GARCH(1,1) -> Symmetric
        print(f"  Fitting Standard GARCH for {target}...")
        model_std = arch_model(returns, p=1, o=0, q=1, vol='Garch', dist='ged')
        res_std = model_std.fit(disp='off')
        
        om_s, al_s, be_s = res_std.params['omega'], res_std.params['alpha[1]'], res_std.params['beta[1]']
        unc_var_s = om_s / (1 - al_s - be_s)
        nic_std = om_s + al_s * (shocks**2) + be_s * unc_var_s
        
        # 2. GJR-GARCH(1,1,1) -> Asymmetric
        print(f"  Fitting GJR-GARCH for {target}...")
        model_gjr = arch_model(returns, p=1, o=1, q=1, vol='Garch', dist='ged')
        res_gjr = model_gjr.fit(disp='off')
        
        om_g = res_gjr.params['omega']
        al_g = res_gjr.params['alpha[1]']
        ga_g = res_gjr.params['gamma[1]']
        be_g = res_gjr.params['beta[1]']
        unc_var_g = om_g / (1 - al_g - ga_g/2 - be_g)
        
        nic_gjr = np.zeros_like(shocks)
        for i, shock in enumerate(shocks):
            indicator = 1 if shock < 0 else 0
            nic_gjr[i] = om_g + (al_g + ga_g * indicator) * (shock**2) + be_g * unc_var_g
            
        # Plotting
        ax.plot(shocks, np.sqrt(nic_gjr), color='#d62728', linewidth=2.5, label='GJR-GARCH (Asymmetric)')
        ax.plot(shocks, np.sqrt(nic_std), color='#1f77b4', linestyle='--', linewidth=2, label='GARCH (Symmetric)')
        
        ax.set_title(f'{target} News Impact Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Past Return Shock (%)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Next-Day Predicted Volatility (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.suptitle("Leverage Effect Analysis across Asset Classes", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    out_file = OUT_DIR / 'news_impact_curve_all.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved Multi-Asset News Impact Curve to {out_file.relative_to(ROOT)}")

if __name__ == "__main__":
    plot_news_impact_curve()
