from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
from scipy.stats import norm
# Import both the AC_PARAMS dictionary and the data loading function
from src.utils.data_loader import AC_PARAMS, load_and_calculate_market_params


# --- 1. MDP State Contract (Used by RL Agent) ---
# Think of it as a snapshot of the trading situation at any moment.

@dataclass(frozen=True)
class OptimalExecutionState:
    # 0: Current Mid-Price (P_t)
    current_price: float
    # 1: Inventory Left (X_t)
    inventory_left: float
    # 2: Time Steps Remaining (T-t)
    time_remaining: int
    # 3: Volatility (Sigma)
    volatility: float
    # 4: Price Trend Vector (N lags of normalized log returns)
    price_trend_vector: np.ndarray
    # 5: Last permanent impact (theta * V_t-1)
    last_perm_impact: float


# The action is a continuous float: percentage of remaining shares to sell
Action = float


# --- 2. Environment Class: Almgren-Chriss Simulator ---

class OptimalExecutionEnv:
    def __init__(self, initial_shares: float, total_time_steps: int, lookback_window: int = 5):

        # Load environment data and calculated parameters (Sigma only)
        self.data_df, self.sigma = load_and_calculate_market_params()

        # Fixed AC model parameters
        self.ETA = AC_PARAMS["TEMPORARY_IMPACT_ETA"]        # 3e-6
        self.THETA = AC_PARAMS["PERMANENT_IMPACT_THETA"]    # 1e-7
        self.LAMBDA = AC_PARAMS["RISK_AVERSION_LAMBDA"]     # 2.0
        self.DELTA_T = AC_PARAMS["TIME_STEP_DELTA"]         # 5.0

        # Episode constraints
        self.X0 = initial_shares        # Initial shares to liquidate
        self.T = total_time_steps       # Total steps for episode
        self.N = lookback_window        # How many historical returns to include
        self.max_index = len(self.data_df) - self.T

        # Internal state tracking
        self.current_step = 0
        self.episode_start_idx = 0
        self.P_REF = None  # Will be set in reset() to episode's starting price

    # Get the last N price movements (log returns) from historical data
    def _get_price_trend(self, idx: int) -> np.ndarray:
        """Extracts N lags of normalized log returns from the loaded data."""
        start = idx - self.N

        if start < 0:
            trend_data = self.data_df['log_return'].iloc[0:idx].values
            # Pad with zeros if less than N steps are available
            return np.pad(trend_data, (self.N - len(trend_data), 0), 'constant')

        trend_data = self.data_df['log_return'].iloc[start:idx].values

        # Handle case where final slice might be short due to index bounds
        if len(trend_data) < self.N:
            return np.pad(trend_data, (self.N - len(trend_data), 0), 'constant')

        return trend_data

    def reset(self) -> OptimalExecutionState:
        """
        Resets the environment for a new episode.

        CRITICAL: P_REF is set to THIS episode's starting price, not a global constant.
        This ensures fair comparison across episodes with different price levels.
        """
        # Select a random starting price index from the historical data
        # Ensure we can run a full episode (T steps) without running off the end of the data
        self.episode_start_idx = np.random.randint(self.N, self.max_index)
        self.current_step = 0

        # Get starting price for this episode
        P0 = self.data_df['close'].iloc[self.episode_start_idx].item()

        # CRITICAL FIX: Set P_REF to this episode's starting price
        # This makes the reward function measure "how well did I execute relative to my entry price"
        # rather than "how close am I to some arbitrary historical price"
        self.P_REF = P0

        initial_history = self._get_price_trend(self.episode_start_idx)

        return OptimalExecutionState(
            current_price=P0,
            inventory_left=self.X0,
            time_remaining=self.T,
            volatility=self.sigma,
            price_trend_vector=initial_history,
            last_perm_impact=0.0
        )

    # CORRECTED: Fixed return type hint with proper tuple contents
    def step(self, state: OptimalExecutionState, action: Action) -> Tuple[OptimalExecutionState, float, bool, Dict]:
        P_t = state.current_price
        X_t = state.inventory_left
        t_rem = state.time_remaining
        sigma = state.volatility

        # 1. Action Validation and Volume Calculation
        A_t = np.clip(action, 0.0, 1.0)  # Ensure action is between 0% and 100%

        # CRITICAL FIX: Force complete liquidation on final step
        if t_rem == 1:
            V_t = X_t  # Sell all remaining inventory on last step
        else:
            V_t = A_t * X_t  # Volume traded in this step

        # --- 2. Execution Price and Revenue (Temporary Impact) ---
        P_exec = P_t - (self.ETA * V_t)
        Revenue_t = V_t * P_exec

        # 3. Next Unperturbed Price (Stochastic Noise / Market Risk)
        # Random Walk component: sigma * sqrt(delta_t) * Z_t (Z_t ~ N(0, 1))
        # Note the use of self.DELTA_T = 5.0 to amplify noise
        noise = sigma * np.sqrt(self.DELTA_T) * norm.rvs()

        # 4. Permanent Price Impact (Liquidity Risk)
        Delta_P_perm = self.THETA * V_t

        # 5. Determine Next Market Price (P_{t+1})
        P_next = P_t + noise - Delta_P_perm

        # 6. Calculate Risk-Averse Reward (Mean-Variance Utility)
        # FIXED: Now uses episode-specific P_REF (set in reset())
        # This measures slippage from the episode's starting price
        risk_penalty = self.LAMBDA * (P_exec - self.P_REF) ** 2
        R_t = Revenue_t - risk_penalty

        # 7. Check Termination
        X_next = X_t - V_t
        t_next = t_rem - 1

        # Episode ends when time runs out or inventory is fully liquidated
        done = (t_next <= 0) or (X_next <= 1e-6)

        # 8. Update State Vector (S_{t+1})
        self.current_step += 1

        # Fetch the next price trend for the augmented belief state
        # Indexing for the history uses the original historical data index plus steps taken
        history_index = self.episode_start_idx + self.current_step
        next_trend = self._get_price_trend(history_index)

        new_state = OptimalExecutionState(
            current_price=P_next,
            inventory_left=X_next,
            time_remaining=t_next,
            volatility=sigma,
            price_trend_vector=next_trend,
            last_perm_impact=Delta_P_perm
        )

        # Return new state, reward, done flag, and info dict (standard Gym API)
        return new_state, R_t, done, {}

# --- End of trading_env.py ---