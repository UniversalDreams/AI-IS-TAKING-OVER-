# scripts/main_ddpg.py
import os, sys

# --- Make sure we can import from src ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.agents.ddpg_tf2 import Agent
from src.utils import plot_learning_curve
from src.envs import OptimalExecutionEnv


# ---- TRAINING THE AGENT ----

def normalize_state_manual(state, env):
    """Normalize state to help neural network training."""
    if not hasattr(env, '_price_stats_cached'):
        env._price_mean = env.data_df['close'].mean()
        env._price_std = env.data_df['close'].std()
        env._typical_impact = env.THETA * (env.X0 / env.T)
        env._price_stats_cached = True

    return np.array([
        (state.current_price - env._price_mean) / env._price_std,
        state.inventory_left / env.X0,
        state.time_remaining / env.T,
        state.volatility / env.sigma,
        *state.price_trend_vector,
        state.last_perm_impact / max(env._typical_impact, 1e-8)
    ], dtype=np.float32)


if __name__ == '__main__':
    env = OptimalExecutionEnv(initial_shares=10000, total_time_steps=50)
    state_dim = 10  # features (price, inventory, etc.)

    agent = Agent(input_dims=[state_dim], env=None, n_actions=1)
    n_games = 2000
    best_score = -np.inf
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            state = env.reset()
            observation = normalize_state_manual(state, env)
            action = np.random.uniform(0, 1)
            state_, reward, done, info = env.step(state, action)
            observation_ = normalize_state_manual(state_, env)
            agent.remember(observation, [action], reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    no_improvement_count = 0
    for i in range(n_games):
        state = env.reset()
        observation = normalize_state_manual(state, env)
        done = False
        score = 0
        terminal_reward = 0.0

        agent.noise = max(0.05, 0.2 * (0.995 ** i))

        while not done:
            action = agent.choose_action(observation, evaluate)
            action_value = float(action[0]) if hasattr(action, '__iter__') else float(action)
            state_, reward, done, info = env.step(state, action_value)
            observation_ = normalize_state_manual(state_, env)

            score += reward
            if done and reward != 0:
                terminal_reward = reward

            agent.remember(observation, [action_value], reward, observation_, done)
            if not load_checkpoint:
                agent.learn()

            observation = observation_
            state = state_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            no_improvement_count = 0
            if not load_checkpoint:
                agent.save_models()
        else:
            no_improvement_count += 1

        if no_improvement_count >= 500:
            print(f"\nEarly stopping at episode {i}: No improvement for 500 episodes")
            print(f"Best avg score achieved: {best_score:.4f}")
            break

        if i < 20 or i % 50 == 0:
            print(f'ep {i:4d} | reward {terminal_reward:7.4f} | avg {avg_score:7.4f} | noise {agent.noise:.3f}')
        else:
            print(f'ep {i:4d} | reward {terminal_reward:7.4f} | avg {avg_score:7.4f}')

    if not load_checkpoint:
        x = [i + 1 for i in range(len(score_history))]
        figure_file = 'plots/optimal_execution.png'
        plot_learning_curve(x, score_history, figure_file)
