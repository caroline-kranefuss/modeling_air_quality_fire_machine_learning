import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import itertools

def check_stationarity(ts, name):
    """Check if time series is stationary"""
    result = adfuller(ts.dropna())
    print(f"\n{name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  P-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print(f"  → Stationary")
        return True
    else:
        print(f"  → Non-Stationary (consider differencing)")
        return False
    

def fit_arima_model(ts, name, order=(1,1,1)):
    """Fit ARIMA model and return fitted model"""
    try:
        # Remove NaN values
        ts_clean = ts.dropna()
        
        if len(ts_clean) < 30:
            print(f"\n{name}: Insufficient data (n={len(ts_clean)})")
            return None
        
        print(f"\n{name}:")
        print(f"  Data points: {len(ts_clean)}")
        print(f"  Mean PM2.5: {ts_clean.mean():.2f}")
        print(f"  Std PM2.5: {ts_clean.std():.2f}")
        
        # Fit ARIMA model
        model = ARIMA(ts_clean, order=order)
        fitted = model.fit()
        
        print(f"  ARIMA{order} AIC: {fitted.aic:.2f}")
        print(f"  AR coefficient(s): {fitted.arparams}")
        print(f"  MA coefficient(s): {fitted.maparams}")
        
        # Calculate recovery metrics
        ar_coef = fitted.arparams[0] if len(fitted.arparams) > 0 else 0
        if ar_coef > 0 and ar_coef < 1:
            half_life = np.log(0.5) / np.log(ar_coef)
            print(f"  Half-life: {half_life:.2f} days")
        else:
            print(f"  Half-life: N/A (AR coefficient = {ar_coef:.3f})")
        
        return fitted
    except Exception as e:
        print(f"\n{name}: Model fitting failed - {str(e)}")
        return None
    
def grid_search_arima(ts, name, max_p=5, max_d=2, max_q=5, seasonal=False):
    """
    Perform grid search over ARIMA parameters
    """
    print(f"\n{name}")
    print("-" * 60)
    
    # Clean data
    ts_clean = ts.dropna()
    
    if len(ts_clean) < 50:
        print(f"Insufficient data: {len(ts_clean)} observations")
        return None
    
    # Define parameter ranges
    p_range = range(0, max_p + 1)
    d_range = range(0, max_d + 1)
    q_range = range(0, max_q + 1)
    
    # Generate all combinations
    orders = list(itertools.product(p_range, d_range, q_range))
    
    # Remove (0,0,0)
    orders = [order for order in orders if order != (0, 0, 0)]
    
    print(f"Testing {len(orders)} model combinations...")
    
    results = []
    
    for order in orders:
        try:
            # Split into train/test (80/20)
            train_size = int(len(ts_clean) * 0.8)
            train, test = ts_clean[:train_size], ts_clean[train_size:]
            
            # Fit model on training data
            model = ARIMA(train, order=order)
            fitted = model.fit()
            
            # Calculate in-sample metrics
            aic = fitted.aic
            bic = fitted.bic
            
            # Calculate out-of-sample RMSE
            try:
                forecast = fitted.forecast(steps=len(test))
                rmse = np.sqrt(np.mean((test.values - forecast.values) ** 2))
            except:
                rmse = np.nan
            
            # Store results
            results.append({
                'order': order,
                'p': order[0],
                'd': order[1],
                'q': order[2],
                'aic': aic,
                'bic': bic,
                'rmse': rmse,
                'params': len(fitted.params)
            })
            
        except Exception as e:
            # Model failed to converge
            continue
    
    if len(results) == 0:
        print("No models converged successfully")
        return None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by AIC
    results_df = results_df.sort_values('aic')
    
    print(f"\nSuccessfully fitted {len(results_df)} models")
    print(f"\nTop 10 models by AIC:")
    print(results_df.head(10)[['order', 'aic', 'bic', 'rmse']].to_string(index=False))
    
    # Get best model
    best_order = results_df.iloc[0]['order']
    
    print(f"\n✓ Best model: ARIMA{best_order}")
    print(f"  AIC: {results_df.iloc[0]['aic']:.2f}")
    print(f"  BIC: {results_df.iloc[0]['bic']:.2f}")
    print(f"  Out-of-sample RMSE: {results_df.iloc[0]['rmse']:.2f}")
    
    return results_df, best_order

def diagnose_arima(ts, order, name):
    """
    Perform comprehensive diagnostics on ARIMA model
    """
    print(f"\n{name} - ARIMA{order}")
    print("-" * 60)
    
    ts_clean = ts.dropna()
    
    # Fit model on full data
    model = ARIMA(ts_clean, order=order)
    fitted = model.fit()
    
    # Get residuals
    residuals = fitted.resid
    
    # 1. Summary statistics
    print("\n1. Residual Statistics:")
    print(f"   Mean: {residuals.mean():.6f} (should be ~0)")
    print(f"   Std Dev: {residuals.std():.4f}")
    print(f"   Min: {residuals.min():.4f}")
    print(f"   Max: {residuals.max():.4f}")
    
    # 2. Normality test
    print("\n2. Normality Test (Jarque-Bera):")
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    print(f"   Test Statistic: {jb_stat:.4f}")
    print(f"   P-value: {jb_pval:.4f}")
    if jb_pval > 0.05:
        print("   ✓ Residuals appear normally distributed (p > 0.05)")
    else:
        print("   ✗ Residuals deviate from normality (p < 0.05)")
    
    # 3. Autocorrelation test (Ljung-Box)
    print("\n3. Autocorrelation Test (Ljung-Box):")
    lb_results = acorr_ljungbox(residuals, lags=10, return_df=True)
    significant_lags = lb_results[lb_results['lb_pvalue'] < 0.05]
    
    if len(significant_lags) == 0:
        print("   ✓ No significant autocorrelation detected (all p > 0.05)")
    else:
        print(f"   ✗ Significant autocorrelation at {len(significant_lags)} lags:")
        print(significant_lags[['lb_stat', 'lb_pvalue']].to_string())
    
    # 4. Heteroskedasticity
    print("\n4. Heteroskedasticity Check:")
    # Split residuals into first and second half
    mid = len(residuals) // 2
    first_half_var = residuals[:mid].var()
    second_half_var = residuals[mid:].var()
    ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)
    print(f"   Variance ratio (first/second half): {ratio:.2f}")
    if ratio < 2:
        print("   ✓ Variance appears stable (ratio < 2)")
    else:
        print("   ✗ Possible heteroskedasticity (ratio >= 2)")
    
    # 5. Parameter significance
    print("\n5. Parameter Significance:")
    print(f"   Coefficients and p-values:")
    for param_name, param_val, pval in zip(fitted.param_names, fitted.params, fitted.pvalues):
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"   {param_name:20s}: {param_val:8.4f}  (p={pval:.4f}) {sig}")
    
    return fitted, residuals

