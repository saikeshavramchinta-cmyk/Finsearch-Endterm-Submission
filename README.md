
Conversation with Gemini
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

    print(top_10[['Strike', 'Type', 'Market_Price', 'BS_Price', 'Percentage_Error']].round(2).to_string(index=False)) give readme file for explaining this code

Here is a complete README.md file designed to explain your code, its requirements, the underlying mathematics, and how to execute it.

You can copy and paste the block below directly into a README.md file in your project directory.

Markdown
# NIFTY Options Black-Scholes-Merton Evaluator

This Python script calculates the theoretical Black-Scholes-Merton (BSM) prices for NIFTY index options and compares them against actual market prices. It calculates accuracy metrics (RMSE, MAE, and Percentage Error) for near-the-money options to help you evaluate how closely the market trades to the theoretical model.

## Features

* **Continuous Dividend Yield:** Adapts the standard Black-Scholes model for an index by factoring in a continuous dividend yield ($q$).
* **Vectorized Pricing:** Uses `NumPy` and `SciPy` for rapid, vectorized calculation of option chains.
* **Automated Data Fallbacks:** Attempts to load historical spot prices from a CSV, but gracefully falls back to pulling live data via `yfinance` if the required date is missing.
* **Smart Filtering:** Automatically filters out deep in-the-money (ITM) anomalies, expired options, and focuses the accuracy report strictly on near-the-money strikes (within 300 points of the spot price).

## Mathematical Model

The script uses the BSM model for European options on an asset paying a continuous dividend yield. 

The theoretical price of a Call ($C$) and Put ($P$) are calculated as:

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

Where:
* $$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma \sqrt{T}}$$
* $$d_2 = d_1 - \sigma \sqrt{T}$$

**Variables used:**
* $S$ = Spot price of the index
* $K$ = Strike price
* $T$ = Time to maturity (in years)
* $r$ = Risk-free interest rate (annualized)
* $q$ = Continuous dividend yield (annualized)
* $\sigma$ = Implied volatility
* $N(\cdot)$ = Cumulative distribution function of the standard normal distribution

## Prerequisites

Ensure you have Python installed along with the following libraries:

```bash
pip install numpy pandas scipy scikit-learn yfinance
Data Requirements
The script expects the following CSV files in the root directory. If the VIX or Yield files are missing or lack matching dates, the script will default to hardcoded values (13.79% for VIX and 7.14% for the risk-free rate).

nifty_options_data.csv: The main options chain dataset.

Expected columns: EXPIRY_DT, TIMESTAMP, OPTION_TYP (CE/PE), STRIKE_PR, CLOSE (or standard Time_To_Maturity, Type, Strike, Market_Price).

nifty_index_data.csv: Historical NIFTY 50 spot prices.

Expected columns: HistoricalDate (or Date), Close.

india_vix_historical.csv: Historical India VIX data used for the σ input.

Expected columns: HistoricalDate (or Date), Close.

india_10yr_yield_historical.csv: Historical 10-year Indian Government Bond yields used for the risk-free rate (r).

Expected columns: HistoricalDate (or Date), Close (or Price).

How it Works
Data Prep: The script loads the options chain and standardizes column names. It calculates the Time to Maturity (DTE) in years and filters down to the nearest expiry date.

Variable Retrieval: It fetches the NIFTY spot price, India VIX, and the 10-year yield for the specific target date found in the options dataset.

Pricing: The get_bsm_prices function executes the mathematical model across the entire dataframe.

Evaluation: The evaluate_accuracy function calculates the intrinsic value to establish valid market boundaries. It filters for options within a 300-point moneyness window and calculates Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

Output: Prints an accuracy report and a top-10 table of the options trading closest to the spot price alongside their percentage deviation from theoretical models.

Usage
Simply run the script from your terminal:

Bash
python option_pricer.py
(Replace option_pricer.py with whatever you named your Python file).

generate a readme file that contains common questions, and explaining the above code

Required Data Files
Ensure the following CSV files are present in the same directory as the script:

nifty_options_data.csv: The main options chain dataset.

nifty_index_data.csv: Historical NIFTY 50 spot prices.

india_vix_historical.csv: Historical India VIX data.

india_10yr_yield_historical.csv: Historical 10-year Indian Government Bond yields.

Common Questions (FAQ)
1. Why does the code use a dividend yield (q)?
Unlike non-dividend-paying stocks where the standard Black-Scholes model applies, stock indices consist of multiple companies that pay dividends throughout the year. The continuous dividend yield accounts for the expected drop in the index value as these dividends are paid out, leading to more accurate theoretical pricing.

2. Why does the script only evaluate options within 300 points of the spot price (Moneyness <= 300)?
Deep In-The-Money (ITM) and deep Out-of-The-Money (OTM) options often suffer from low liquidity, wide bid-ask spreads, and stale pricing. By focusing on Near-The-Money and At-The-Money (ATM) options, the script ensures that the calculated accuracy metrics (RMSE/MAE) reflect the model's actual performance against active, reliable market data.

3. What happens if my CSV data is missing the target date?
The script is built with graceful fallbacks:

Spot Price: It will attempt to download the exact required date's spot price using yfinance (^NSEI). If that fails, it falls back to the most recent date available in your CSV.

VIX and Yield: It defaults to a hardcoded assumption (13.79% for Volatility, 7.14% for Risk-Free Rate) to ensure the script doesn't crash and still provides an estimation.

4. How is Intrinsic Value used to filter anomalies?
The intrinsic value of a Call is Spot - Strike (or Strike - Spot for Puts), with a minimum of 0. An option's market price should theoretically never be less than its intrinsic value. The script uses the filter df['Market_Price'] >= (df['Intrinsic_Value'] - 0.5) to drop rows where bad data or extreme illiquidity has caused the market price to fall below this floor (allowing a 0.5 point margin of error).

5. Why do the theoretical and market prices sometimes differ significantly?
The Black-Scholes model assumes constant volatility and log-normal distribution of returns. In reality, markets exhibit "volatility smiles" or "skews" where OTM options have higher implied volatilities due to tail-risk pricing. The script uses a single fixed volatility (India VIX), which will inherently cause divergences, especially as you move further from the money.
"""

with open('README.md', 'w') as f:
f.write(markdown_content)

print("README generated successfully.")


```text?code_stdout&code_event_index=1
README generated successfully.

Your README file is ready
MD icon
README
 MD 
Open

I have generated the README.md file based on your provided code. It includes an overview of how the code works, data requirements, and a detailed "Common Questions (FAQ)" section that explains the rationale behind dividend yields, data fallbacks, intrinsic value filtering, and moneyness thresholds.




Gemini is AI and can make mistakes.

# NIFTY Options Black-Scholes-Merton (BSM) Evaluator

This project contains a Python script designed to calculate the theoretical Black-Scholes-Merton (BSM) prices for NIFTY index options, compare them against actual market prices, and evaluate the model's accuracy.

## How the Code Works

The script is divided into two primary logical components: the mathematical calculation and the data evaluation pipeline.

### 1. Mathematical Calculation (`get_bsm_prices`)
The script uses the Merton extension of the Black-Scholes model, which incorporates a continuous dividend yield (`q`). This is essential for pricing index options like the NIFTY 50, as the underlying constituent stocks pay dividends.
* **Inputs:** Spot Price (S), Strike Price (K), Time to Maturity (T), Risk-free Rate (r), Dividend Yield (q), and Implied Volatility (sigma).
* **Process:** It calculates the `d1` and `d2` probabilities using vectorization (via NumPy) for high-speed processing across the entire options chain. It then computes Call and Put prices based on the option type.

### 2. Data Pipeline and Evaluation (`evaluate_accuracy` & `__main__`)
* **Data Prep:** The script loads options data, standardizes datetime columns, calculates the Time to Expiration (DTE), and filters out expired options.
* **Variable Extraction:** 
    * **Spot Price:** Extracted from a local CSV or fetched live via `yfinance` if the date is missing.
    * **Volatility (VIX) & Risk-Free Rate:** Loaded from historical CSVs or defaults to standard assumptions (13.79% for VIX, 7.14% for yield) if data is missing.
* **Filtering:** Calculates Intrinsic Value to filter out anomalous market prices (prices trading below intrinsic value) and restricts the accuracy evaluation to near-the-money options (within 300 points of the spot price).
* **Metrics:** Computes Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Percentage Error for the filtered dataset using `scikit-learn`.

## Prerequisites

Install the required Python libraries before running the script:
```bash
pip install numpy pandas scipy scikit-learn yfinance
```

## Required Data Files
Ensure the following CSV files are present in the same directory as the script:
* `nifty_options_data.csv`: The main options chain dataset.
* `nifty_index_data.csv`: Historical NIFTY 50 spot prices.
* `india_vix_historical.csv`: Historical India VIX data.
* `india_10yr_yield_historical.csv`: Historical 10-year Indian Government Bond yields.

---

## Common Questions (FAQ)

### 1. Why does the code use a dividend yield (`q`)?
Unlike non-dividend-paying stocks where the standard Black-Scholes model applies, stock indices consist of multiple companies that pay dividends throughout the year. The continuous dividend yield accounts for the expected drop in the index value as these dividends are paid out, leading to more accurate theoretical pricing.

### 2. Why does the script only evaluate options within 300 points of the spot price (Moneyness <= 300)?
Deep In-The-Money (ITM) and deep Out-of-The-Money (OTM) options often suffer from low liquidity, wide bid-ask spreads, and stale pricing. By focusing on Near-The-Money and At-The-Money (ATM) options, the script ensures that the calculated accuracy metrics (RMSE/MAE) reflect the model's actual performance against active, reliable market data.

### 3. What happens if my CSV data is missing the target date?
The script is built with graceful fallbacks:
* **Spot Price:** It will attempt to download the exact required date's spot price using `yfinance` (`^NSEI`). If that fails, it falls back to the most recent date available in your CSV.
* **VIX and Yield:** It defaults to a hardcoded assumption (13.79% for Volatility, 7.14% for Risk-Free Rate) to ensure the script doesn't crash and still provides an estimation.

### 4. How is Intrinsic Value used to filter anomalies?
The intrinsic value of a Call is `Spot - Strike` (or `Strike - Spot` for Puts), with a minimum of 0. An option's market price should theoretically never be less than its intrinsic value. The script uses the filter `df['Market_Price'] >= (df['Intrinsic_Value'] - 0.5)` to drop rows where bad data or extreme illiquidity has caused the market price to fall below this floor (allowing a 0.5 point margin of error).

### 5. Why do the theoretical and market prices sometimes differ significantly?
The Black-Scholes model assumes constant volatility and log-normal distribution of returns. In reality, markets exhibit "volatility smiles" or "skews" where OTM options have higher implied volatilities due to tail-risk pricing. The script uses a single fixed volatility (India VIX), which will inherently cause divergences, especially as you move further from the money.
README.md
Displaying README.md.
