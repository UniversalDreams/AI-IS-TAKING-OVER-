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

        # Validate inputs
        if initial_shares <= 0:
            raise ValueError(f"initial_shares must be positive, got {initial_shares}")
        if total_time_steps <= 0:
            raise ValueError(f"total_time_steps must be positive, got {total_time_steps}")
        if lookback_window <= 0:
            raise ValueError(f"lookback_window must be positive, got {lookback_window}")

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

        self.episode_metrics = {
            'total_revenue': 0.0,
            'total_penalty': 0.0,
            'total_volume': 0.0,
            'avg_slippage': 0.0,
            'total_implementation_shortfall': 0.0,
            'execution_prices': [],
            'actions': []
        }

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
        # risk_penalty = self.LAMBDA * (P_exec - self.P_REF) ** 2
        # Use slippage
        relative_slippage = (P_exec - self.P_REF) / self.P_REF
        risk_penalty = self.LAMBDA * (relative_slippage ** 2) * 10000

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

        # Calculate implementation shortfall for tracking
        implementation_shortfall = V_t * (self.P_REF - P_exec)

        # Calculate slippage in basis points
        slippage_bps = ((P_exec - self.P_REF) / self.P_REF) * 10000

        # Enhanced info dictionary
        info = {
            # Execution metrics
            'volume': V_t,
            'volume_pct': V_t / self.X0,
            'execution_price': P_exec,
            'slippage': P_exec - self.P_REF,
            'slippage_bps': slippage_bps,
            'implementation_shortfall': implementation_shortfall,

            # Market impact
            'temporary_impact': self.ETA * V_t,
            'permanent_impact': Delta_P_perm,

            # State tracking
            'inventory_remaining_pct': X_next / self.X0,
            'time_progress': 1 - (t_next / self.T),

            # Reward components
            'revenue': Revenue_t,
            'risk_penalty': risk_penalty,

            # Episode tracking
            'step': self.current_step,
            'forced_liquidation': t_rem == 1
        }

        # Update episode metrics
        self.episode_metrics['total_revenue'] += Revenue_t
        self.episode_metrics['total_penalty'] += risk_penalty
        self.episode_metrics['total_volume'] += V_t
        self.episode_metrics['execution_prices'].append(P_exec)
        self.episode_metrics['actions'].append(action)

        if done:
            # Compute episode-level metrics
            self.episode_metrics['avg_slippage'] = (
                    np.mean(self.episode_metrics['execution_prices']) - self.P_REF
            )
            info['episode_metrics'] = self.episode_metrics.copy()

        # Return new state, reward, done flag, and info dict (standard Gym API)
        return new_state, R_t, done, {}


def get_twap_policy(self) -> float:
    """
    Returns the TWAP (Time-Weighted Average Price) action.

    TWAP simply sells equal amounts each step: 1/T of remaining inventory.
    This is the industry baseline to beat.
    """
    return 1.0 / self.T


def run_twap_episode(self) -> Dict:
    """
    Run one episode using TWAP strategy for benchmarking.

    Returns episode metrics.
    """
    state = self.reset()
    total_reward = 0
    metrics = {
        'rewards': [],
        'prices': [],
        'volumes': [],
        'slippages': []
    }

    while True:
        action = self.get_twap_policy()
        state, reward, done, info = self.step(state, action)

        total_reward += reward
        metrics['rewards'].append(reward)
        metrics['prices'].append(info['execution_price'])
        metrics['volumes'].append(info['volume'])
        metrics['slippages'].append(info['slippage_bps'])

        if done:
            break

    metrics['total_reward'] = total_reward
    return metrics

# --- End of trading_env.py ---