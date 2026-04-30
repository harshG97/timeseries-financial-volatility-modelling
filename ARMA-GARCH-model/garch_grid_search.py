
import pandas as pd
import numpy as np
from arch import arch_model
import itertools
import os
import warnings

warnings.filterwarnings("ignore")

def garch_grid_search(target, freq, exog):
    """
    Performs a grid search for an ARMA-GJR-GARCH model.
    """
    path = f'data/splits/{freq}/{exog}/{target}/train.csv'
    data = pd.read_csv(path, index_col='date', parse_dates=True)
    
    # Use 'ret' column as the endogenous variable
    returns = data['ret'] * 100  # Scale returns for better convergence

    # Define the grid for p, q, o
    p_range = range(1, 4)
    q_range = range(1, 4)
    o_range = range(0, 2)

    best_aic = np.inf
    best_model = None
    best_params = {}

    # Grid search
    for p, o, q in itertools.product(p_range, o_range, q_range):
        try:
            model = arch_model(returns, p=p, o=o, q=q, vol='Garch', dist='ged')
            res = model.fit(disp='off')
            if res.aic < best_aic:
                best_aic = res.aic
                best_model = res
                best_params = {'p': p, 'o': o, 'q': q}
        except Exception as e:
            continue

    return {
        'target': target,
        'freq': freq,
        'exog': exog,
        'best_p': best_params.get('p'),
        'best_o': best_params.get('o'),
        'best_q': best_params.get('q'),
        'aic': best_aic,
    }

if __name__ == '__main__':
    targets = ['SPY', 'OIL', 'GOLD']
    freqs = ['daily', 'weekly']
    exogs = ['no_exog', 'with_exog']

    results = []
    for target in targets:
        for freq in freqs:
            for exog in exogs:
                print(f"Running for {target}, {freq}, {exog}...")
                result = garch_grid_search(target, freq, exog)
                results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv('garch_grid_search_results.csv', index=False)
    print("Grid search complete. Results saved to garch_grid_search_results.csv")
