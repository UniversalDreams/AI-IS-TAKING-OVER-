import sys
import os
#----TRAINING THE AGENT----

# Store original directory for imports
original_dir = os.path.dirname(os.path.abspath(__file__))

# Add Environment directory to path (use local copy with all files)
env_path = os.path.join(original_dir, 'Environment ')
sys.path.insert(0, env_path)
sys.path.insert(0, original_dir)

# Change to Environment directory so data loading works
os.chdir(env_path)

import numpy as np
from ddpg_tf2 import Agent
#graphs of how agent performance improves over training episodes
from utils import plot_learning_curve
from src.envs.trading_env import OptimalExecutionEnv

## Helper function for state normalization (matches partner's implementation)
def normalize_state_manual(state, env):
    """Normalize state to help neural network training - matches root env logic."""
    # Calculate price statistics from data if not cached
    if not hasattr(env, '_price_stats_cached'):
        env._price_mean = env.data_df['close'].mean()
        env._price_std = env.data_df['close'].std()
        # Typical impact per step (average volume per step × permanent impact factor)
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

##Import the environment you want to use
if __name__ == '__main__':
    # Initialize custom environment with parameters
    # 50 steps × 5 min/step = 250 minutes (~4 hours of trading)
    # More realistic horizon for learning optimal liquidation strategy
    env = OptimalExecutionEnv(initial_shares=10000, total_time_steps=50)
    
    # State dimension: price + inventory + time + volatility + trend_vector + last_impact
    #we have the 6 features we need but the trend vector has a length of 5
    state_dim = 10
    
    agent = Agent(input_dims=[state_dim], env=None,  # env=None since custom env
            n_actions=1)  # 1 action: percentage to sell
    # Reduced episodes for hackathon with early stopping protection
    n_games = 2000

    best_score = -np.inf  # Start with very low score
    score_history = []
    load_checkpoint = False  # Train from scratch with protections
    #load saved progress
    if load_checkpoint: #this is to sample from a randomized batch 
        n_steps = 0
        while n_steps <= agent.batch_size: #making sure memory isn't empty
            state = env.reset()
            observation = normalize_state_manual(state, env)  
            action = np.random.uniform(0, 1)  # Random action between 0 and 1
            state_, reward, done, info = env.step(state, action)
            observation_ = normalize_state_manual(state_, env)  
            agent.remember(observation, [action], reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False
    #training of the agent
    no_improvement_count = 0  # Track episodes without improvement
    for i in range(n_games): #num of episosdes 
        state = env.reset() #initial state at the start of each episode
        observation = normalize_state_manual(state, env)  # Matches partner's normalization 
        done = False
        score = 0
        terminal_reward = 0.0
        
        # Decay exploration noise over time (0.2 → 0.05)
        agent.noise = max(0.05, 0.2 * (0.995 ** i))
        
        while not done:
            action = agent.choose_action(observation, evaluate)
            action_value = float(action[0]) if hasattr(action, '__iter__') else float(action)
            
            state_, reward, done, info = env.step(state, action_value) #next state after action
            observation_ = normalize_state_manual(state_, env)  # Matches partner's normalization
            
            score += reward
            if done and reward != 0:  # Terminal reward
                terminal_reward = reward
            agent.remember(observation, [action_value], reward, observation_, done)
            #don't want to train during evaluation 
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            state = state_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            no_improvement_count = 0  # Reset counter
            #only save modals during training 
            if not load_checkpoint:
                agent.save_models()
        else:
            no_improvement_count += 1
            
        # Early stopping if no improvement for 500 episodes (was too aggressive at 300)
        if no_improvement_count >= 500:
            print(f"\nEarly stopping at episode {i}: No improvement for 500 episodes")
            print(f"Best avg score achieved: {best_score:.4f}")
            break

        # Print with terminal reward info
        if i < 20 or i % 50 == 0:  # More frequent early logging
            print(f'ep {i:4d} | reward {terminal_reward:7.4f} | avg {avg_score:7.4f} | noise {agent.noise:.3f}')
        else:
            print(f'ep {i:4d} | reward {terminal_reward:7.4f} | avg {avg_score:7.4f}')

    if not load_checkpoint:
        # Use actual number of episodes trained (not n_games) for plotting
        x = [i+1 for i in range(len(score_history))]
        figure_file = 'plots/optimal_execution.png'
        plot_learning_curve(x, score_history, figure_file)

