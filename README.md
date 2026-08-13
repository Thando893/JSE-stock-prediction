# JSE Stock Next-Day Prediction

A data science project testing whether next-day price movements for JSE-listed
stocks can be predicted from historical price and volume data alone, using
technical indicators as features.

**Headline finding:** across two stocks from different sectors (Naspers —
tech/media, Standard Bank — banking), models show no reliable, meaningful
edge over naive baselines. This is consistent with the weak-form efficient
market hypothesis and a well-replicated result in academic finance research.

## Why this matters

A lot of "stock prediction" projects online report suspiciously high accuracy
(70-80%+) on this exact type of task. That's almost always a sign of data
leakage — usually a random (rather than chronological) train/test split, or
features/scaling fit on the full dataset before splitting. This project
deliberately uses a strict chronological split and reports the honest,
unglamorous result.

## Methodology

- **Data**: Daily OHLCV via [yfinance](https://pypi.org/project/yfinance/)
- **Features** (11, all scale-free — ratios/percentages, not raw price levels):
  lagged returns (1/2/3/5-day), price relative to 5-day and 20-day moving
  averages, 10-day rolling volatility, 14-day RSI, volume % change, 5-day
  volume moving average, high-low spread
- **Targets**:
  - Regression: next-day return (%)
  - Classification: next-day direction (up/down)
- **Split**: chronological — trained on 2018–2024, tested on 2025–2026 (no
  shuffling, no lookahead)
- **Models**: Linear Regression & XGBoost (regression); Logistic Regression,
  Random Forest & XGBoost (classification), each benchmarked against a naive
  baseline

## Results

### Naspers (NPN.JO) — tech/media

| Model | Direction Accuracy |
|---|---|
| Naive (majority class) | 51.2% |
| Logistic Regression | 48.8% |
| Random Forest | 53.0% |
| XGBoost | 52.2% |

| Model | Return R² |
|---|---|
| Naive (0% return) | 0.0000 |
| Linear Regression | -0.0036 |
| XGBoost | -0.0092 |

### Standard Bank (SBK.JO) — banking

| Model | Direction Accuracy |
|---|---|
| Naive (majority class) | 54.2% |
| Logistic Regression | 47.3% |
| Random Forest | 56.0% |
| XGBoost | 51.2% |

*(Standard Bank's higher naive baseline reflects a class imbalance in the test
period — the stock simply closed up more often than down — not greater
predictability.)*

### Cross-stock takeaway

| | Naspers | Standard Bank |
|---|---|---|
| Best model vs. baseline gap | +1.8 pts (Random Forest) | +1.8 pts (Random Forest) |
| Return regression R² | Negative for both models | *(see notes below)* |

Both stocks show the same pattern: a small, inconsistent edge that's well
within the range you'd expect from noise, not a reliable signal. Random
Forest edges out the baseline in both cases by a similar margin — but
Logistic Regression *underperforms* the baseline in both cases too. This
inconsistency across model types, rather than a consistent win, is itself
evidence against a real exploitable signal.

## Repo structure

```
jse-stock-prediction/
├── data/                     # Cleaned historical price/volume data
├── src/
│   └── predict.py            # Reusable pipeline — works for any JSE ticker
├── results/                  # Prediction outputs, charts
├── report/                   # Full written report (docx)
└── README.md
```

## Usage

```bash
pip install pandas numpy scikit-learn xgboost yfinance matplotlib

# Using a local CSV
python src/predict.py --ticker NPN.JO --csv data/naspers_jse_clean.csv

# Or download fresh data directly
python src/predict.py --ticker SBK.JO
```

## Key methodological notes

- **Chronological split, not random.** Randomly shuffling time-series data
  before splitting leaks future information into training and produces
  artificially inflated performance — a common and serious error in stock
  prediction projects.
- **Predicting return, not absolute price.** An early version of this
  analysis attempted to predict absolute next-day closing price directly
  from scale-free features, which failed badly (the model had no anchor to
  the current price level, which drifted substantially over the sample
  period). Predicting return instead — the standard approach in the
  literature — corrected this.
- **Negative R² is a real, informative result.** It means the model performs
  *worse* than simply guessing "no change" — not a bug, but a sign the
  features carry no exploitable relationship with next-day returns at this
  horizon.

## Next steps

- Extend to a longer prediction horizon (5-day or 20-day trend)
- Add cross-asset features (USD/ZAR, Tencent as a Naspers leading indicator,
  sector peers)
- Add news/sentiment features
- Walk-forward (rolling) validation instead of a single fixed split
- If used for real investment decisions: benchmark any edge against
  transaction costs and slippage — a small statistical edge is not
  automatically a profitable trading strategy

## Data source

Yahoo Finance via [yfinance](https://pypi.org/project/yfinance/).
