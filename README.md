# GreekFire Walk-Forward Backtesting Engine

GreekFire is a high-performance stock trading backtesting engine written in Julia. It utilizes a walk-forward testing methodology to train machine learning models (Random Forest and Linear Regression) on historical data to pick the best-performing stock from a curated universe.

## Features
- **Walk-Forward Testing**: Expanding window training to eliminate lookahead bias.
- **Technical Indicators**: Manually implemented RSI, MACD, Bollinger Bands, SMA ratios, ATR, Volume ratios, and Momentum.
- **Synthetic Greeks**: Black-Scholes implementation for Delta, Gamma, Theta, and Vega.
- **Parallel Execution**: Native Julia threading for fast backtesting.
- **Octave Integration**: Automatic export of results for visualization in GNU Octave.

## Setup

### Prerequisites
- [Julia](https://julialang.org/downloads/) (1.11+ recommended)
- [GNU Octave](https://gnu.org/software/octave/) (for visualizations)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd GreekFire
   ```

## Running the Engine

Execute the full pipeline with a single command:
```bash
julia --project=. main.jl
```

This command will:
1. Fetch historical data for the stock universe and SPY.
2. Compute all technical and Greeks features.
3. Run the parallelized walk-forward backtest (2009 - Present).
4. Print overall performance metrics to the console.
5. Export results as CSVs to the `/octave` directory.

## Visualizing Results

To generate the performance charts, navigate to the `/octave` directory and run:
```bash
octave plot_all.m
```

## Project Structure
- `main.jl`: Pipeline entry point.
- `data_loader.jl`: Data fetching and caching.
- `features.jl`: Indicator and Greek computations.
- `model.jl`: ML model training and prediction.
- `backtest.jl`: Walk-forward loop implementation.
- `metrics.jl`: Performance statistic calculations.
- `visualizer.jl`: CSV export for Octave.
- `/data`: Cached market data.
- `/results`: Walk-forward logs.
- `/octave`: Exported data and plotting scripts.
