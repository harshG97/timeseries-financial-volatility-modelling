
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.graphics.tsaplots import plot_acf
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

def validate_and_analyze_garch(target, freq, exog, p, o, q):
    """
    Fits the best GARCH model on the training data, evaluates on the validation
    set, and performs residual analysis.
    """
    train_path = f'data/splits/{freq}/{exog}/{target}/train.csv'
    val_path = f'data/splits/{freq}/{exog}/{target}/val.csv'

    train_data = pd.read_csv(train_path, index_col='date', parse_dates=True)
    val_data = pd.read_csv(val_path, index_col='date', parse_dates=True)

    train_returns = train_data['ret'] * 100
    val_returns = val_data['ret'] * 100

    # 1. Fit the model on the training data
    model = arch_model(train_returns, p=p, o=o, q=q, vol='Garch', dist='ged')
    res = model.fit(disp='off')

    # 2. Forecast on the validation set
    forecasts = res.forecast(horizon=len(val_returns), start=0, reindex=False)
    forecast_variance = forecasts.variance.iloc[-1] # Get the last row of forecasts

    # Align forecast with validation data
    forecast_variance = forecast_variance.reindex(val_returns.index).ffill()
    
    # Calculate out-of-sample MSE
    val_mse = np.mean((val_returns - forecast_variance.values)**2)

    # 3. Residual Analysis
    std_resid = res.std_resid
    
    # Create output directory
    output_dir = f'modeling/validation_analysis/{target}/{freq}/{exog}'
    os.makedirs(output_dir, exist_ok=True)

    # Plot Standardized Residuals
    plt.figure(figsize=(12, 6))
    std_resid.plot(title='Standardized Residuals')
    plt.savefig(f'{output_dir}/standardized_residuals.png')
    plt.close()

    # Plot ACF of Standardized Residuals
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(std_resid, lags=40, ax=ax, title='ACF of Standardized Residuals')
    plt.savefig(f'{output_dir}/acf_standardized_residuals.png')
    plt.close()

    # Plot ACF of Squared Standardized Residuals
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(std_resid**2, lags=40, ax=ax, title='ACF of Squared Standardized Residuals')
    plt.savefig(f'{output_dir}/acf_squared_standardized_residuals.png')
    plt.close()

    return {
        'target': target,
        'freq': freq,
        'exog': exog,
        'val_mse': val_mse,
        'log_likelihood': res.loglikelihood,
        'aic': res.aic,
        'bic': res.bic
    }

if __name__ == '__main__':
    grid_search_results = pd.read_csv('modeling/garch_grid_search_results.csv')
    
    validation_results = []
    
    for _, row in grid_search_results.iterrows():
        target = row['target']
        freq = row['freq']
        exog = row['exog']
        p = int(row['best_p'])
        o = int(row['best_o'])
        q = int(row['best_q'])
        
        print(f"Validating and analyzing for {target}, {freq}, {exog} with p={p}, o={o}, q={q}...")
        
        try:
            result = validate_and_analyze_garch(target, freq, exog, p, o, q)
            validation_results.append(result)
        except Exception as e:
            print(f"  Failed for {target}/{freq}/{exog}: {e}")
            continue
            
    validation_df = pd.DataFrame(validation_results)
    validation_df.to_csv('modeling/garch_validation_results.csv', index=False)
    
    print("\\nValidation and analysis complete.")
    print("Results saved to modeling/garch_validation_results.csv")
    print("Residual plots saved in modeling/validation_analysis/")
