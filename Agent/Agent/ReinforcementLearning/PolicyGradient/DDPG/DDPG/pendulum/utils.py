import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curve(x, scores, figure_file):
    """Plot the learning curve showing episode scores over time."""
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, running_avg, label='Running Average (100 episodes)', linewidth=2)
    plt.plot(x, scores, alpha=0.3, label='Episode Score')
    plt.title('DDPG Training Progress - Optimal Execution')
    plt.xlabel('Episode')
    plt.ylabel('Score (Revenue - Risk Penalty)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figure_file)
    print(f'Plot saved to {figure_file}')
