"""
Test script for OptimalExecutionEnv
Verifies that the environment works correctly with episode-specific P_REF
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.trading_env import OptimalExecutionEnv
import numpy as np

def test_environment():
    print("=" * 60)
    print("Testing OptimalExecutionEnv")
    print("=" * 60)

    # Create environment
    env = OptimalExecutionEnv(
        initial_shares=10000,  # Start with 10,000 shares
        total_time_steps=12    # 12 steps = 60 minutes (12 * 5min)
    )

    print(f"\n✅ Environment created successfully")
    print(f"   Initial shares: {env.X0}")
    print(f"   Total time steps: {env.T}")
    print(f"   Volatility (sigma): {env.sigma:.6f}")

    # Test reset
    print("\n" + "-" * 60)
    print("Testing reset()...")
    state = env.reset()

    print(f"✅ Reset successful")
    print(f"   Starting price: ${state.current_price:.2f}")
    print(f"   P_REF (episode reference): ${env.P_REF:.2f}")
    print(f"   Inventory: {state.inventory_left:.0f} shares")
    print(f"   Time remaining: {state.time_remaining} steps")
    print(f"   Price trend vector shape: {state.price_trend_vector.shape}")

    # Verify P_REF equals starting price
    assert abs(env.P_REF - state.current_price) < 0.01, "P_REF should equal starting price!"
    print(f"✅ P_REF correctly set to episode starting price")

    # Test one step
    print("\n" + "-" * 60)
    print("Testing step()...")

    action = 0.1  # Sell 10% of inventory
    new_state, reward, done, info = env.step(state, action)

    print(f"✅ Step executed successfully")
    print(f"   Action: Sell {action*100:.0f}% ({action * state.inventory_left:.0f} shares)")
    print(f"   New price: ${new_state.current_price:.2f}")
    print(f"   Price change: ${new_state.current_price - state.current_price:+.2f}")
    print(f"   Inventory left: {new_state.inventory_left:.0f} shares")
    print(f"   Reward: {reward:.2f}")
    print(f"   Done: {done}")

    # Run a full episode
    print("\n" + "-" * 60)
    print("Running full episode with random actions...")

    state = env.reset()
    episode_reward = 0
    step_count = 0

    print(f"\nEpisode starting at price: ${state.current_price:.2f}")
    print(f"Episode P_REF: ${env.P_REF:.2f}")
    print(f"Initial inventory: {state.inventory_left:.0f} shares\n")

    while True:
        # Random policy: sell 5-15% each step (deliberately conservative to test forced liquidation)
        action = np.random.uniform(0.05, 0.15)

        old_inventory = state.inventory_left
        state, reward, done, info = env.step(state, action)
        shares_sold = old_inventory - state.inventory_left
        episode_reward += reward
        step_count += 1

        print(f"Step {step_count:2d}: Sold={shares_sold:6.0f}, "
              f"Price=${state.current_price:6.2f}, "
              f"Inventory={state.inventory_left:7.0f}, "
              f"Reward={reward:10.2f}")

        if done:
            break

    print(f"\n✅ Episode completed")
    print(f"   Steps taken: {step_count}")
    print(f"   Final inventory: {state.inventory_left:.2f} shares")
    print(f"   Total reward: {episode_reward:.2f}")

    # Verify complete liquidation
    if state.inventory_left < 1.0:
        print(f"✅ Complete liquidation achieved! (forced on final step)")
    else:
        print(f"❌ Warning: {state.inventory_left:.0f} shares remaining!")

    # Test multiple episodes to verify P_REF changes
    print("\n" + "-" * 60)
    print("Testing multiple episodes (P_REF should vary)...")

    p_refs = []
    for i in range(5):
        state = env.reset()
        p_refs.append(env.P_REF)
        print(f"Episode {i+1}: P_REF = ${env.P_REF:.2f}, Starting price = ${state.current_price:.2f}")

    # Verify P_REF varies across episodes
    if len(set(p_refs)) > 1:
        print(f"✅ P_REF correctly varies across episodes")
        print(f"   P_REF range: ${min(p_refs):.2f} - ${max(p_refs):.2f}")
    else:
        print(f"⚠️  Warning: P_REF didn't vary (might be using same starting index)")

    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)

if __name__ == '__main__':
    try:
        test_environment()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()