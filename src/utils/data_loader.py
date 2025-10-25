import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict

# --- 1. Fixed Market Microstructure Parameters (Optimized for NVIDIA) ---
AC_PARAMS: Dict[str, float] = {
    "PERMANENT_IMPACT_THETA": 1e-7,  # Low (NVDA very liquid)
    "TEMPORARY_IMPACT_ETA": 3e-6,  # Low-moderate (tight spreads)
    "RISK_AVERSION_LAMBDA": 2.0,  # Conservative starting point
    "TIME_STEP_DELTA": 5.0  # Trade every 5 minutes
}

DATA_PATH = os.path.join("data", "NVDA_historical.csv")


def load_and_calculate_market_params() -> Tuple[pd.DataFrame, float]:
    """
    Loads historical price data and calculates sigma.

    NOTE: P_REF (reference price) is NOT calculated here because it should be
    episode-specific (set to each episode's starting price), not a global constant.

    Returns: (DataFrame with price data, Estimated Volatility Sigma)
    """
    print(f"Loading environment data from {DATA_PATH} to calculate sigma...")
    try:
        df = pd.read_csv(DATA_PATH, index_col='timestamp', parse_dates=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Simulation data not found at {DATA_PATH}. Check fetch script run."
        )

    # Calculate Log Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Clean up NA values created by shift(1)
    df.dropna(inplace=True)

    # Calculate Volatility (Sigma): Standard deviation of log returns
    # Sigma is the core parameter for the A-C stochastic noise
    volatility_sigma = df['log_return'].std().item()

    print(f"Market Volatility Sigma calculated: {volatility_sigma:.8f}")

    return df, volatility_sigma


if __name__ == '__main__':
    try:
        # Unpack two return values:
        data_df, sigma_val = load_and_calculate_market_params()

        print(f"\nTotal Trading Steps Loaded: {len(data_df)}")
        print(f"Calculated Sigma: {sigma_val:.8f}")
        print(f"\nA-C Parameters:")
        for key, value in AC_PARAMS.items():
            print(f"  {key}: {value}")

        # Show sample price range for context
        print(f"\nPrice Range in Dataset:")
        print(f"  Min: ${data_df['close'].min():.2f}")
        print(f"  Max: ${data_df['close'].max():.2f}")
        print(f"  First: ${data_df['close'].iloc[0]:.2f}")
        print(f"  Last: ${data_df['close'].iloc[-1]:.2f}")

    except Exception as e:
        print(f"Validation failed: {e}")