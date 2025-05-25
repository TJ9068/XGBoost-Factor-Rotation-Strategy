# XGBoost-Factor-Rotation-Strategy

A macro-factor-enhanced machine learning strategy using Gradient Boosting to dynamically allocate weights across Fama-French 5 factors. This model integrates macroeconomic signals and momentum features to construct a market-neutral, alpha-generating portfolio.

##  Overview

This project implements a Gradient Boosted Decision Tree (XGBoost) model to forecast the relative performance of Fama-French 5 factors based on lagged macroeconomic indicators and short-term factor momentum. The strategy dynamically allocates daily weights to factor portfolios to maximize risk-adjusted returns.

- **Objective**: Generate stable alpha with low volatility and market neutrality
- **Tooling**: Python, XGBoost, Pandas, NumPy, Scikit-learn
- **Data**: Fama-French 5 factors, macroeconomic indicators (VIX, Yield Spread, Put/Call Ratio, Expected Inflation, Consumer Sentiment)

## Methodology

### 1. Feature Engineering
- **Macroeconomic inputs** (lagged 5/15 days):
  - VIX Index (`VIX_lag5`)
  - Put/Call Ratio (`PCR_lag5`)
  - Yield Spread (`YldSpread_lag5`)
  - Expected Inflation (`ExpInfl_lag5`)
  - Consumer Sentiment (`ConsSent_lag15`)

- **Momentum features**:
  - 20-day rolling averages of Fama-French 5 factor returns

### 2. Model Architecture
- Model: **XGBoost Regressor**
- Target: Daily future return of each Fama-French factor
- Output: Predicted return → transformed to **portfolio weights** via softmax/normalization
- Daily rebalancing with risk controls (volatility thresholds, exposure caps)

### 3. Risk Management
- Position limits on daily volatility and max drawdown
- Near-zero **beta exposure** to SPX, ensuring market neutrality
- Daily risk monitoring using rolling volatility and drawdown filters

## Results

| Metric | Value |
|--------|-------|
| **Annualized Return** | 11.37% |
| **Volatility** | 2.15% |
| **Sharpe Ratio** | 5.28 |
| **Max 1-Day Drawdown** | -0.46% |
| **Portfolio Beta (SPX)** | 0.0063 |
| **Alpha vs SPX** | 9.04% |
| **Alpha vs FF5 Model** | 7.71% |

The strategy significantly outperforms both the market and a benchmark equal-weighted factor portfolio, while remaining uncorrelated to broader market movements.

## Risks and Considerations

- Model overfitting and performance decay
- Structural breaks in macro–factor relationships
- Transaction costs, slippage, and liquidity issues
- Correlation risk between input features
