import os, sys, numpy as np

# make 'src' importable when you run from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
from src.agents.ddpg_torch import Agent
from src.utils.plot_utils import plot_learning_curve

# If Box2D is installed, you can use LunarLanderContinuous-v3.
# Otherwise switch to Pendulum-v1 (no Box2D) and adjust dims/actions accordingly.
env = gym.make('LunarLanderContinuous-v3')

agent = Agent(alpha=0.000025, beta=0.00025,
              input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = []
for i in range(1000):
    obs, info = env.reset()
    done = False
    score = 0.0

    while not done:
        act = agent.choose_action(obs)
        new_state, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()

        score += reward
        obs = new_state

    score_history.append(score)
    print(f'episode {i:4d} | score {score:7.2f} | trailing100 {np.mean(score_history[-100:]):.3f}')

x = list(range(1, len(score_history) + 1))
plot_learning_curve(x, score_history, 'plots/LunarLander-alpha000025-beta00025-400-300.png')
