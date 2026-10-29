# PSU Banks Time-Series Data Analysis Pipeline

Pure-R data analysis pipeline for 12 NSE-listed Public Sector Undertaking (PSU) banks. Fetches daily OHLCV from Yahoo Finance, cleans and aligns the panel, computes returns, generates EDA plots, and calculates a comprehensive set of technical indicators for every bank.

---

## Ticker Universe

12 PSU bank stocks (all `.NS` — NSE):

| Ticker | Bank |
|--------|------|
| SBIN.NS | State Bank of India |
| BANKBARODA.NS | Bank of Baroda |
| PNB.NS | Punjab National Bank |
| CANBK.NS | Canara Bank |
| UNIONBANK.NS | Union Bank of India |
| INDIANB.NS | Indian Bank |
| BANKINDIA.NS | Bank of India |
| CENTRALBK.NS | Central Bank of India |
| MAHABANK.NS | Bank of Maharashtra |
| IOB.NS | Indian Overseas Bank |
| UCOBANK.NS | UCO Bank |
| PSB.NS | Punjab & Sind Bank |

**Date coverage:** 2019-01-01 to 2026-10-28 (1,933 trading days). All 12 banks have full history.

---

## Pipeline Steps

All scripts are pure R and auto-install required packages at the top.

### Step 0 — Data Fetching
**File:** `r_analysis/01_fetch_data.R`
- Pulls daily OHLCV from Yahoo Finance via `quantmod::getSymbols`
- Saves one raw CSV per ticker to `data/`
- Writes `_fetch_log.csv` with fetch status and row counts

### Step 1 — Cleaning & Alignment
**File:** `r_analysis/02_clean_align.R`
- Reads raw CSVs, aligns all series to a common date index
- Computes daily log-returns: `log(Close_t / Close_{t-1})`
- Filters stocks with < 6 months of data
- Outputs:
  - `cleaned/close_prices_aligned.csv` — 1,933 dates × 12 tickers
  - `cleaned/daily_log_returns.csv` — aligned log-returns
  - `cleaned/summary_stats.csv` — per-ticker metrics (ann. return, vol, Sharpe, max drawdown)
  - `cleaned/*_ohlcv.csv` — 12 individual cleaned files

### Step 2 — Exploratory Data Analysis
**File:** `r_analysis/03_eda.R`
- Normalised price chart — all 12 banks rebased to 100 (log scale Y-axis)
- Return distribution histograms — 12 tickers with mean + SD annotations
- Rolling volatility — 30-day annualized vol for major PSU banks
- Outputs 3 PNGs to `plots/`

### Step 3 — Technical Indicators
**File:** `r_analysis/04_indicators.R`
- Computes **9 indicator families** (20 total columns) for **every** bank:

| Indicator | Purpose | Parameters |
|-----------|---------|------------|
| SMA | Trend | 20 / 50 / 200 day |
| EMA | Faster trend | 12 / 26 day |
| RSI | Momentum | 14-day |
| MACD | Trend + momentum | 12, 26, 9 |
| Bollinger Bands | Volatility envelope | 20-day, 2σ |
| ATR | Average volatility | 14-day |
| VWAP | Institutional price benchmark | 20-day rolling |
| Stochastic Oscillator | Momentum | 14-3-3 |
| OBV | Volume pressure | cumulative |

- Saves one `*_indicators.csv` per ticker to `indicators/` (12 files)
- Generates one **5-panel comprehensive chart** per ticker to `plots/` (12 PNGs)
  - Panel 1: Price + SMAs + VWAP + Bollinger Bands
  - Panel 2: RSI(14) + Stochastic %K
  - Panel 3: MACD + Signal + Histogram
  - Panel 4: ATR(14)
  - Panel 5: OBV

---

## How to Run

The R scripts are shared by both universes and live in `r_analysis/` at the repo root.
Run them from anywhere, passing the universe:

```bash
Rscript r_analysis/01_fetch_data.R psu
Rscript r_analysis/02_clean_align.R psu
Rscript r_analysis/03_eda.R psu
Rscript r_analysis/04_indicators.R psu
```

Outputs are written to this folder. Ticker lists are in `r_analysis/config.R`.

Each script is self-contained and installs missing R packages automatically via `install.packages(..., repos = "https://cloud.r-project.org/")`.

---

## Directory Structure

```
Data_analysis_psu/
|-- README.md                    # This file
|-- company_to_ticker_map.csv    # Company name -> ticker mapping
|-- data/                        # 12 raw OHLCV CSVs from Yahoo
|-- cleaned/
|   |-- close_prices_aligned.csv
|   |-- daily_log_returns.csv
|   |-- summary_stats.csv
|   |-- *_ohlcv.csv             # 12 individual cleaned files
|-- indicators/                  # 12 *_indicators.csv (20 cols each)
|-- plots/
    |-- 01_normalised_prices.png
    |-- 02_return_distributions.png
    |-- 03_rolling_volatility.png
    |-- 04_indicators_*.png      # 12 comprehensive indicator charts
```

---

## Key Summary Statistics

From `cleaned/summary_stats.csv` (sorted by Sharpe ratio):

| Ticker | Trading Days | Ann. Return % | Ann. Vol % | Sharpe | Max DD % |
|--------|-------------|---------------|------------|--------|----------|
| SBIN.NS | 1,932 | 15.15 | 31.08 | **0.49** | -59.49 |
| MAHABANK.NS | 1,932 | 21.48 | 44.31 | **0.48** | -56.25 |
| INDIANB.NS | 1,932 | 15.73 | 43.39 | **0.36** | -85.21 |
| CANBK.NS | 1,932 | 9.63 | 40.04 | **0.24** | -74.33 |
| UNIONBANK.NS | 1,932 | 9.50 | 41.80 | **0.23** | -76.87 |
| BANKBARODA.NS | 1,932 | 8.23 | 38.77 | **0.21** | -74.10 |
| IOB.NS | 1,932 | 8.61 | 46.32 | **0.19** | -63.99 |
| PNB.NS | 1,932 | 5.35 | 38.12 | **0.14** | -72.94 |
| BANKINDIA.NS | 1,932 | 2.91 | 41.68 | **0.07** | -71.19 |
| UCOBANK.NS | 1,932 | -0.35 | 43.00 | **-0.01** | -68.38 |
| CENTRALBK.NS | 1,932 | -3.28 | 45.89 | **-0.07** | -72.62 |
| PSB.NS | 1,932 | -7.34 | 46.99 | **-0.16** | -75.77 |

**Notable findings:**
- **SBI** has the best risk-adjusted profile (Sharpe 0.49): 15.15% annualised return at 31.08% volatility, the lowest in the set
- **Bank of Maharashtra (MAHABANK.NS)** has the highest return (21.48% annualised, Sharpe 0.48) but with much higher volatility
- **Indian Bank (INDIANB.NS)** had a severe max drawdown of -85.21% despite a 15.73% annualised return, indicating extreme tail risk
- **UCO Bank**, **Central Bank of India** and **Punjab & Sind Bank** are the only negative-return banks over the full period
- All PSU banks showed extreme COVID volatility spikes (100%+ annualized) in March 2020, with recovery trajectories varying significantly

---

## Notes

- **Pure R pipeline** — all core analysis scripts written in R with auto-install headers
- **Full-history panel** — all 12 banks have complete 1,933-day histories, so every indicator (including SMA200) is fully populated
- **Gappy series handling** — internal NAs are forward-filled (`na.locf`) before indicator computation so TTR rolling functions don't crash

---

## Planned Next Steps (not yet implemented)

| Step | Description |
|------|-------------|
| 05_decomposition.R | STL trend / seasonal / residual decomposition |
| 06_stationarity.R | ADF tests, ACF/PACF plots |
| 07_correlation.R | Return correlation matrix + hierarchical clustering |
| 08_risk.R | VaR, CVaR, max drawdown, Sortino ratio |
