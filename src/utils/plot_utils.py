import os
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    """Plot the learning curve showing episode scores over time."""
    running_avg = np.zeros(len(scores), dtype=np.float32)
    for i in range(len(running_avg)):
        start = max(0, i - 100)
        running_avg[i] = np.mean(scores[start:(i + 1)])

    # Create containing directory if needed
    out_dir = os.path.dirname(figure_file) or "."
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(x, running_avg, label='Running Average (100 episodes)', linewidth=2)
    plt.plot(x, scores, alpha=0.3, label='Episode Score')
    plt.title('DDPG Training Progress - Optimal Execution')
    plt.xlabel('Episode')
    plt.ylabel('Score (Revenue - Risk Penalty)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figure_file, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {figure_file}')
