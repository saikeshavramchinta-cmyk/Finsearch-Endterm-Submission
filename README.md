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

### Results are shown in the Keshav_Video in the google drive link(https://drive.google.com/drive/folders/1LXAu1MvBG5appw6WjTpXIGoXUDEQWOUR?usp=sharing)

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
The Black-Scholes model assumes constant volatility and log-normal distribution of returns. In reality, markets exhibit "volatility smiles" or "skews" where OTM options have higher implied volatilities due to tail-risk pricing. The script uses a single fixed volatility (India VIX), which will inherently cause divergences, especially as you move further from the money
