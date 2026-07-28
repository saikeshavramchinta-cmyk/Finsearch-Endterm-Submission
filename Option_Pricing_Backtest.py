import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf

def get_bsm_prices(S, K, T, r, q, sigma, option_types):
    """
    Calculates Black-Scholes-Merton prices for a vectorized set of options.
    Includes continuous dividend yield (q) since we're working with an index.
    """
    # Prevent division by zero for options expiring today
    T = np.maximum(T, 1e-6) 
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_prices = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return np.where(option_types == 'call', call_prices, put_prices)

def evaluate_accuracy(df, spot, r, q, sigma):
    # calculate theoretical prices
    df['BS_Price'] = get_bsm_prices(
        spot, df['Strike'], df['Time_To_Maturity'], 
        r, q, sigma, df['Type']
    )
    
    # figure out intrinsic value to filter out deep ITM weirdness
    df['Intrinsic_Value'] = np.where(
        df['Type'] == 'call',
        np.maximum(spot - df['Strike'], 0),
        np.maximum(df['Strike'] - spot, 0)
    )
    
    df['Moneyness'] = np.abs(df['Strike'] - spot)
    
    # filter for valid market prices and options close to ATM (within 300 pts)
    valid_mask = df['Market_Price'] >= (df['Intrinsic_Value'] - 0.5)
    near_money_mask = df['Moneyness'] <= 300
    
    filtered_df = df[valid_mask & near_money_mask].copy()

    # run the stats
    mse = mean_squared_error(filtered_df['Market_Price'], filtered_df['BS_Price'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(filtered_df['Market_Price'], filtered_df['BS_Price'])
    
    filtered_df['Percentage_Error'] = (
        np.abs(filtered_df['BS_Price'] - filtered_df['Market_Price']) / filtered_df['Market_Price']
    ) * 100
    
    print("\n--- Accuracy Report (ATM/Near-Money) ---")
    print(f"Sample size: {len(filtered_df)} options")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return filtered_df

if __name__ == "__main__":
    # NIFTY div yield assumption
    div_yield = 0.0135  
    
    # 1. Load and prep the options data
    try:
        options = pd.read_csv("nifty_options_data.csv")
    except Exception as e:
        print(f"Couldn't read options data. Error: {e}")
        exit()
    
    # standardize datetime columns
    if 'EXPIRY_DT' in options.columns:
        options['EXPIRY_DT'] = pd.to_datetime(options['EXPIRY_DT'])
    if 'TIMESTAMP' in options.columns:
        options['TIMESTAMP'] = pd.to_datetime(options['TIMESTAMP'])
        
    # calculate DTE if it's missing
    if 'Time_To_Maturity' not in options.columns and all(c in options.columns for c in ['EXPIRY_DT', 'TIMESTAMP']):
        options['Time_To_Maturity'] = (options['EXPIRY_DT'] - options['TIMESTAMP']).dt.days / 365.0
        
    # clean up column names and types
    if 'OPTION_TYP' in options.columns:
        options['Type'] = options['OPTION_TYP'].map({'CE': 'call', 'PE': 'put'})
    if 'STRIKE_PR' in options.columns:
        options = options.rename(columns={'STRIKE_PR': 'Strike', 'CLOSE': 'Market_Price'})
    if 'SYMBOL' in options.columns:
        options = options[options['SYMBOL'] == 'NIFTY'].copy()
        
    # drop expired/invalid rows
    options = options[options['Time_To_Maturity'] > 0].copy()
    options = options.dropna(subset=['Strike', 'Time_To_Maturity', 'Type', 'Market_Price']).reset_index(drop=True)
    
    # filter down to the nearest expiry
    if 'EXPIRY_DT' in options.columns:
        nearest_expiry = options['EXPIRY_DT'].min()
        options = options[options['EXPIRY_DT'] == nearest_expiry].copy()

    if 'TIMESTAMP' not in options.columns:
        print("Missing TIMESTAMP column. Aborting.")
        exit()
        
    target_date = options['TIMESTAMP'].iloc[0]
    print(f"\nPricing options for date: {target_date.date()}")

    # 2. Grab the spot price
    current_spot = None
    try:
        index_data = pd.read_csv("nifty_index_data.csv")
        date_col = 'HistoricalDate' if 'HistoricalDate' in index_data.columns else 'Date'
        index_data[date_col] = pd.to_datetime(index_data[date_col])
        
        # trim to our target date
        history = index_data[index_data[date_col].dt.date <= target_date.date()].sort_values(by=date_col).reset_index(drop=True)
        last_date_in_csv = history[date_col].iloc[-1].date()
        
        # fallback to yfinance if the CSV is missing the date we actually need
        if last_date_in_csv != target_date.date():
            print(f"Warning: CSV is missing spot data for {target_date.date()}. Trying yfinance...")
            
            # fetch just the window we need
            start_str = target_date.strftime('%Y-%m-%d')
            end_str = (target_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            yf_data = yf.download("^NSEI", start=start_str, end=end_str, progress=False)
            
            if not yf_data.empty:
                current_spot = float(np.squeeze(yf_data['Close'].iloc[0]))
                print(f"Success: Grabbed spot price from web ({current_spot:.2f})")
            else:
                print(f"Web fetch failed too. Falling back to stale spot from {last_date_in_csv}. Expect errors.")
                current_spot = float(history['Close'].iloc[-1])
        else:
            current_spot = float(history['Close'].iloc[-1])
            print(f"Loaded spot price from CSV: {current_spot:.2f}")
            
    except Exception as e:
        print(f"Failed to get spot price: {e}")
        exit()

    # 3. Grab implied volatility (India VIX)
    try:
        vix_data = pd.read_csv("india_vix_historical.csv") 
        vix_col = 'HistoricalDate' if 'HistoricalDate' in vix_data.columns else 'Date'
        vix_data[vix_col] = pd.to_datetime(vix_data[vix_col])
        
        vix_match = vix_data[vix_data[vix_col].dt.date == target_date.date()]
        
        if not vix_match.empty:
            volatility = float(vix_match['Close'].iloc[0]) / 100.0
            print(f"Loaded India VIX: {volatility:.4f}")
        else:
            print("VIX date mismatch. Using 13.79% default.")
            volatility = 0.1379 
    except Exception:
        print("Couldn't read VIX dataset. Using 13.79% default.")
        volatility = 0.1379

    # 4. Grab risk-free rate (10yr yield)
    try:
        yield_data = pd.read_csv("india_10yr_yield_historical.csv") 
        
        # fix Investing.com 'Price' header if it's there
        if 'Price' in yield_data.columns and 'Close' not in yield_data.columns:
            yield_data = yield_data.rename(columns={'Price': 'Close'})
            
        yield_col = 'HistoricalDate' if 'HistoricalDate' in yield_data.columns else 'Date'
        yield_data[yield_col] = pd.to_datetime(yield_data[yield_col])
        
        yield_match = yield_data[yield_data[yield_col].dt.date == target_date.date()]
        
        if not yield_match.empty:
            risk_free_rate = float(yield_match['Close'].iloc[0]) / 100.0
            print(f"Loaded risk-free rate: {risk_free_rate:.4f}")
        else:
            print("Yield date mismatch. Using 7.14% default.")
            risk_free_rate = 0.0714 
    except Exception:
        print("Couldn't read yield dataset. Using 7.14% default.")
        risk_free_rate = 0.0714

    # 5. Run the pricing and print results
    results = evaluate_accuracy(options, current_spot, risk_free_rate, div_yield, volatility)
    
    top_10 = results.sort_values(by=['Moneyness', 'Type']).head(10)
    
    print("\nSnippet of options closest to the money:")
    print(top_10[['Strike', 'Type', 'Market_Price', 'BS_Price', 'Percentage_Error']].round(2).to_string(index=False))
