# NBFC Time-Series Data Analysis Pipeline

Pure-R data analysis pipeline for 31 NSE-listed Non-Banking Financial Companies (NBFCs). Fetches daily OHLCV from Yahoo Finance, cleans and aligns the panel, computes returns, generates EDA plots, and calculates a comprehensive set of technical indicators for every stock.

---

## Ticker Universe

31 NBFC stocks (mostly `.NS`, one `.BO` exception):

| Ticker | Ticker | Ticker | Ticker |
|--------|--------|--------|--------|
| ABCAPITAL.NS | AFSL.NS | BAIDFIN.NS | BAJFINANCE.NS |
| CHOLAFIN.NS | CREDITACC.NS | EDELWEISS.NS | FEDFINA.NS |
| FIVESTAR.NS | FUSION.NS | IIFL.NS | INDOSTAR.NS |
| JIOFIN.NS | LTF.NS | M&MFIN.NS | MANAPPURAM.NS |
| MASFIN.NS | MONEYBOXX.NS | MUTHOOTFIN.NS | MUTHOOTMF.NS |
| NIYOGIN.BO | PAISALO.NS | PFS.NS | PNBGILTS.NS |
| POONAWALLA.NS | SBFC.NS | SBICARD.NS | SHRIRAMFIN.NS |
| SUNDARMFIN.NS | TATACAP.NS | UGROCAP.NS |  |

**Date coverage:** 2019-01-01 to 2026-10-28 (1,933 trading days for full-history tickers). Newer listings have partial history only.

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
  - `cleaned/close_prices_aligned.csv` — 1,933 dates × 31 tickers
  - `cleaned/daily_log_returns.csv` — aligned log-returns
  - `cleaned/summary_stats.csv` — per-ticker metrics (ann. return, vol, Sharpe, max drawdown)
  - `cleaned/*_ohlcv.csv` — 31 individual cleaned files

### Step 2 — Exploratory Data Analysis
**File:** `r_analysis/03_eda.R`
- Normalised price chart — all 31 tickers rebased to 100 (log scale Y-axis)
- Return distribution histograms — 12 top tickers with mean + SD annotations
- Rolling volatility — 30-day annualized vol for major NBFCs
- Outputs 3 PNGs to `plots/`

### Step 3 — Technical Indicators
**File:** `r_analysis/04_indicators.R`
- Computes **9 indicator families** (20 total columns) for **every** stock:

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

- Saves one `*_indicators.csv` per ticker to `indicators/` (31 files)
- Generates one **5-panel comprehensive chart** per ticker to `plots/` (31 PNGs)
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
Rscript r_analysis/01_fetch_data.R nbfc
Rscript r_analysis/02_clean_align.R nbfc
Rscript r_analysis/03_eda.R nbfc
Rscript r_analysis/04_indicators.R nbfc
```

Outputs are written to this folder. Ticker lists are in `r_analysis/config.R`.

Each script is self-contained and installs missing R packages automatically via `install.packages(..., repos = "https://cloud.r-project.org/")`.

---

## Directory Structure

```
Data_analysis_nbfc/
|-- README.md                    # This file
|-- data/                        # 31 raw OHLCV CSVs from Yahoo
|-- cleaned/
|   |-- close_prices_aligned.csv
|   |-- daily_log_returns.csv
|   |-- summary_stats.csv
|   |-- *_ohlcv.csv             # 31 individual cleaned files
|-- indicators/                  # 31 *_indicators.csv (20 cols each)
|-- plots/
    |-- 01_normalised_prices.png
    |-- 02_return_distributions.png
    |-- 03_rolling_volatility.png
    |-- 04_indicators_*.png      # 31 comprehensive indicator charts
```

---

## Key Summary Statistics

From `cleaned/summary_stats.csv` (top 5 by Sharpe ratio):

| Ticker | Trading Days | Ann. Return % | Ann. Vol % | Sharpe | Max DD % |
|--------|-------------|---------------|------------|--------|----------|
| MUTHOOTFIN.NS | 1,932 | 20.10 | 35.56 | **0.57** | -46.53 |
| CHOLAFIN.NS | 1,932 | 23.70 | 42.59 | **0.56** | -64.37 |
| BAJFINANCE.NS | 1,932 | 16.16 | 35.84 | **0.45** | -62.52 |
| PAISALO.NS | 1,932 | 21.32 | 50.17 | **0.43** | -68.79 |
| ABCAPITAL.NS | 1,932 | 16.55 | 38.59 | **0.43** | -64.32 |

**Notable findings:**
- MUTHOOTFIN.NS delivered the best risk-adjusted return (Sharpe 0.57) with a relatively mild max drawdown of -46.53%
- POONAWALLA.NS had a severe drawdown of -90.34% despite a decent 16.89% annualized return
- MONEYBOXX.NS and TATACAP.NS (about one year of history each) and NIYOGIN.BO show negative annualized returns
- All major NBFCs exhibited extreme volatility spikes (120-180% annualized) during the March 2020 COVID crash

---

## Notes

- **Pure R pipeline** — all core analysis scripts written in R with auto-install headers
- **Micro-cap handling** — tickers with sparse data (`TATACAP.NS` 149 rows, `MONEYBOXX.NS` 144 rows, `NIYOGIN.BO` 347 rows) still get indicator CSVs and plots, but long-window indicators (SMA200, MACD, BBands) will be all-NA for the initial period
- **Gappy series** — internal NAs are forward-filled (`na.locf`) before indicator computation so TTR rolling functions don't crash, preserving the structural consistency of all 31 output files

---

## Planned Next Steps (not yet implemented)

| Step | Description |
|------|-------------|
| 05_decomposition.R | STL trend / seasonal / residual decomposition |
| 06_stationarity.R | ADF tests, ACF/PACF plots |
| 07_correlation.R | Return correlation matrix + hierarchical clustering |
| 08_risk.R | VaR, CVaR, max drawdown, Sortino ratio |
