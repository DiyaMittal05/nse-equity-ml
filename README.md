# NSE Equity ML

Machine-learning research on Indian equities:

| Folder | What it does | Language |
|---|---|---|
| `xgb_nse_trading/` | 11-step pipeline on 1-minute (next 5 minutes) or daily (next day) bars: OHLCV + FII/DII flows → technical features → XGBoost → walk-forward backtest → SHAP failure analysis → report | Python |
| `r_analysis/` | Shared R scripts (fetch → clean → EDA → indicators) that build the two data folders below | R |
| `Data_analysis_psu/` | Daily OHLCV, cleaned panel, EDA plots and technical indicators for 12 PSU banks | data |
| `Data_analysis_nbfc/` | The same for 31 NBFCs | data |
| `sentiment_agent/` | LangGraph agent that scores a news headline's impact on PSU and NBFC stocks (moderation → routing → macro policy → sentiment → synthesis) | Python |
| `finbert_sentiment/` | Interactive FinBERT (ProsusAI/finbert) headline sentiment classifier; optional backend for the agent | Python |

## How the pieces connect

```
r_analysis/ ──▶ Data_analysis_psu/, Data_analysis_nbfc/

Data_analysis_psu/ ─┐  daily OHLCV + indicators
Data_analysis_nbfc/ ┼─▶ sentiment_agent/ (reads ../Data_analysis_*/ via src/config/settings.py)
                    └─▶ xgb_nse_trading/ (src/csv_loader.py reads the daily OHLCV CSVs)

finbert_sentiment/ ──▶ FinBERT, optional stage A of sentiment_agent/

Yahoo 1-minute ──▶ xgb_nse_trading/ --bars minute (+ FII flows)
```

`sentiment_agent/` and `xgb_nse_trading/` find the R outputs by looking for `Data_analysis_psu/` and
`Data_analysis_nbfc/` next to themselves, so keep those folder names as they are.

## Running

### XGBoost pipeline

Two bar sizes, same 11 steps:

```bash
cd xgb_nse_trading
pip install -r requirements.txt
python main.py --symbol SBIN.NS --bars minute   # 1-minute bars, predicts the next 5 minutes
python main.py --symbol SBIN.NS --bars daily    # daily bars since 2019, predicts the next day's close
```

| | `--bars minute` | `--bars daily` |
|---|---|---|
| Prices | Yahoo Finance 1-minute bars (no key), last ~30 days | `Data_analysis_psu/`, `Data_analysis_nbfc/` (2019-01-01 to 2026-10-28) |
| Label | return over the next 5 minutes, never across the overnight gap | next day's return |
| Features | 1/5/15-minute returns, session VWAP, time of day (session 09:15–15:30), indicators, FII | 1/5/20-day returns, 20-day VWAP, calendar, indicators, FII |
| FII | previous day's figures; the CSV covers every session, so FII features are on | skipped: the CSV covers ~3% of days since 2019 |
| Backtest | one trade every 5 minutes, 1-day gap, 3 bps per position change | one trade a day, 1-day gap, 10 bps |

Yahoo only keeps 1-minute bars for 30 days, so every run merges the download into
`data/raw/ohlcv/minute/<symbol>_1m.csv`. Running it every couple of weeks builds a longer history.
FII figures must cover the same sessions: refresh `data/raw/fii/fii_dii_historical.csv` with the
latest month (new rows are added, old ones kept). Results go to `data/results/<bars>/`.

How the evaluation avoids look-ahead:

- Each bar uses the **previous** day's FII figures (they are published after the close); FII trend and
  3-day mean are computed per FII day, not per bar.
- Early stopping uses the last 15% of the training data; the test set is only used for reporting.
- The walk-forward backtest uses 5 expanding windows with a 1-day gap, scales features on each
  window's training data only, scores non-overlapping trades and charges costs on every position change.
- Features are scale-free (returns, ratios to price).
- Data checks run after every step and stop the run on bad prices, misaligned labels or train/test overlap.

Step 9 and every walk-forward window run SHAP and compare attributions on bars where the model called the
direction right vs wrong (`failure_diagnosis.csv`, `walk_forward_shap.csv`).

Tests: `cd xgb_nse_trading && python -m pytest tests`.

FII features switch on automatically when the FII CSV covers at least 80% of the bars being modelled.

`src/data_loader.py` (Angel One SmartAPI, 1-minute bars) is an alternative minute source if you get
an Angel One key; `main.py` uses Yahoo. Credentials go in `xgb_nse_trading/.env` (see `.env.example`).

### Sentiment agent

```bash
cd sentiment_agent
uv sync
uv run python -m src.main "RBI hikes repo rate by 25 bps"
```

Headline sentiment uses the bundled TF-IDF model. It is trained on template-generated headlines, so
treat its accuracy figure as optimistic. To use FinBERT instead: `uv sync --extra finbert` and set
`SENTIMENT_BACKEND=finbert` (first run downloads ~440 MB). FinBERT handles general finance language
well but misreads Indian banking terms: "PNB NPAs jump to 12%" and "RBI hikes repo rate" both score positive.

### FinBERT

```bash
cd finbert_sentiment
uv run python main.py
```

### R analysis

```bash
Rscript r_analysis/02_clean_align.R nbfc     # psu or nbfc; 01_fetch_data.R refreshes prices from Yahoo
```

See `Data_analysis_psu/README.md` and `Data_analysis_nbfc/README.md`.
