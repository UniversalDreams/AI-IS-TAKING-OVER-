This project implements a Deep Deterministic Policy Gradient (DDPG) agent that learns to optimally liquidate large stock positions by minimizing trading costs while balancing execution speed against market impact. The agent uses an actor-critic neural network architecture trained on a custom Almgren-Chriss trading environment that simulates real market dynamics including temporary and permanent price impact, volatility, and execution risk. We used a 10-dimensional state space to capture price, inventory, time remaining, volatility, price trends, and last market impact, while the agent outputs continuous actions representing the percentage of remaining shares to sell at each 5-minute interval. Through 2000 training episodes with experience replay and decaying exploration noise, the agent learns sophisticated strategies including dynamic position sizing, volatility adaptation, and implementation shortfall minimization compared to naive execution baselines. 

## How to Run
cd Agent/Agent/ReinforcementLearning/PolicyGradient/DDPG/DDPG/pendulum
python main_ddpg.py

The agent will train for up to 2000 episodes with early stopping and save trained models to `tmp/ddpg/`. Training progress and performance curves are saved to `plots/optimal_execution.png`.

## Visual Representation 
https://calhacksfinance.netlify.app/

*Framework inspired by "Foundations of Reinforcement Learning with Applications in Finance" by Ashwin Rao and Tikhon Jelvis* 
