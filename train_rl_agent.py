#!/usr/bin/env python3
"""Train RL agent for adaptive UX optimization."""

import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.rl.environment import CheckoutEnvironment
from src.models.rl.a2c_agent import A2CAgent
from src.config.model_config import RL_CONFIG


def load_personas():
    """Load generated personas."""
    print("=" * 70)
    print("Loading Personas")
    print("=" * 70)
    
    persona_path = Path('data/synthetic/personas/all_personas.pkl')
    
    if not persona_path.exists():
        print(f"❌ Personas not found at {persona_path}")
        print("   Please generate personas first:")
        print("   python3 generate_personas.py")
        return None
    
    with open(persona_path, 'rb') as f:
        personas = pickle.load(f)
    
    print(f"✓ Loaded {len(personas)} personas")
    
    # Count by type
    asd_count = sum(1 for p in personas if p.persona_type == 'ASD')
    adhd_count = sum(1 for p in personas if p.persona_type == 'ADHD')
    nt_count = sum(1 for p in personas if p.persona_type == 'NT')
    
    print(f"  • ASD: {asd_count}")
    print(f"  • ADHD: {adhd_count}")
    print(f"  • NT: {nt_count}")
    
    return personas


def initialize_environment(personas):
    """Initialize the checkout environment."""
    print("\n" + "=" * 70)
    print("Initializing RL Environment")
    print("=" * 70)
    
    env = CheckoutEnvironment(personas)
    
    print(f"✓ Environment created")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Action dimension: {env.action_dim}")
    print(f"  Max steps per episode: {env.max_steps}")
    
    print(f"\n  Available actions:")
    print(f"    0: Reduce animations")
    print(f"    1: Increase whitespace")
    print(f"    2: Simplify steps")
    print(f"    3: Extend timeout")
    print(f"    4: Toggle slow mode")
    
    return env


def initialize_agent():
    """Initialize the A2C agent."""
    print("\n" + "=" * 70)
    print("Initializing A2C Agent")
    print("=" * 70)
    
    agent = A2CAgent(
        state_dim=RL_CONFIG.state_dim,
        action_dim=RL_CONFIG.action_dim,
        hidden_dim=RL_CONFIG.hidden_dim,
        lr_actor=RL_CONFIG.lr_actor,
        lr_critic=RL_CONFIG.lr_critic,
        gamma=RL_CONFIG.gamma
    )
    
    print(f"✓ Agent initialized")
    print(f"  State dim: {RL_CONFIG.state_dim}")
    print(f"  Action dim: {RL_CONFIG.action_dim}")
    print(f"  Hidden dim: {RL_CONFIG.hidden_dim}")
    print(f"  Learning rate (actor): {RL_CONFIG.lr_actor}")
    print(f"  Learning rate (critic): {RL_CONFIG.lr_critic}")
    print(f"  Discount factor (gamma): {RL_CONFIG.gamma}")
    
    return agent


def train_agent(agent, env, num_episodes=100):
    """Train the RL agent."""
    print("\n" + "=" * 70)
    print(f"Training RL Agent ({num_episodes} episodes)")
    print("=" * 70)
    
    episode_rewards = []
    episode_steps = []
    episode_completions = []
    episode_abandonments = []
    
    print("\nStarting training...")
    print("(This will take a few minutes)\n")
    
    for episode in range(num_episodes):
        total_reward, steps = agent.train_episode(env)
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Get episode info
        state, info = env.reset()
        completed = 0
        abandoned = 0
        
        # Run episode to get stats
        for _ in range(env.max_steps):
            action, _ = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            
            if info.get('completed', False):
                completed = 1
            if info.get('abandoned', False):
                abandoned = 1
            
            if terminated or truncated:
                break
        
        episode_completions.append(completed)
        episode_abandonments.append(abandoned)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            completion_rate = np.mean(episode_completions[-10:])
            abandonment_rate = np.mean(episode_abandonments[-10:])
            
            print(f"Episode {episode+1:3d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Steps: {avg_steps:4.1f} | "
                  f"Completion: {completion_rate:.1%} | "
                  f"Abandonment: {abandonment_rate:.1%}")
    
    print("\n✓ Training complete!")
    
    return {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'completions': episode_completions,
        'abandonments': episode_abandonments
    }


def evaluate_agent(agent, env, num_episodes=20):
    """Evaluate the trained agent."""
    print("\n" + "=" * 70)
    print(f"Evaluating Agent ({num_episodes} episodes)")
    print("=" * 70)
    
    eval_rewards = []
    eval_completions = []
    eval_abandonments = []
    eval_cognitive_loads = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        cognitive_loads = []
        
        for step in range(env.max_steps):
            action, _ = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            cognitive_loads.append(info.get('cognitive_load', 0))
            
            if terminated or truncated:
                eval_completions.append(1 if info.get('completed', False) else 0)
                eval_abandonments.append(1 if info.get('abandoned', False) else 0)
                break
        
        eval_rewards.append(total_reward)
        eval_cognitive_loads.append(np.mean(cognitive_loads))
    
    # Calculate metrics
    avg_reward = np.mean(eval_rewards)
    completion_rate = np.mean(eval_completions)
    abandonment_rate = np.mean(eval_abandonments)
    avg_cognitive_load = np.mean(eval_cognitive_loads)
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Completion Rate: {completion_rate:.1%}")
    print(f"  Abandonment Rate: {abandonment_rate:.1%}")
    print(f"  Avg Cognitive Load: {avg_cognitive_load:.2f}")
    
    return {
        'avg_reward': avg_reward,
        'completion_rate': completion_rate,
        'abandonment_rate': abandonment_rate,
        'avg_cognitive_load': avg_cognitive_load
    }


def save_results(agent, training_history, eval_results):
    """Save training results."""
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path('results/rl')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save agent
    agent.save(output_dir / 'rl_agent.pth')
    print(f"✓ Saved agent to {output_dir / 'rl_agent.pth'}")
    
    # Save training history
    with open(output_dir / 'training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)
    print(f"✓ Saved training history")
    
    # Save evaluation results
    with open(output_dir / 'eval_results.pkl', 'wb') as f:
        pickle.dump(eval_results, f)
    print(f"✓ Saved evaluation results")
    
    # Save text summary
    with open(output_dir / 'rl_training_summary.txt', 'w') as f:
        f.write("RL Agent Training Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training Episodes: {len(training_history['rewards'])}\n")
        f.write(f"Final Avg Reward: {np.mean(training_history['rewards'][-10:]):.2f}\n\n")
        f.write("Evaluation Results:\n")
        for key, value in eval_results.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"✓ Saved summary to {output_dir / 'rl_training_summary.txt'}")
    
    return output_dir


def plot_training_curves(training_history, output_dir):
    """Plot training curves."""
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rewards
    axes[0, 0].plot(training_history['rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Steps
    axes[0, 1].plot(training_history['steps'])
    axes[0, 1].set_title('Episode Steps')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Completion rate
    window = 10
    completion_ma = np.convolve(training_history['completions'], 
                                np.ones(window)/window, mode='valid')
    axes[1, 0].plot(completion_ma)
    axes[1, 0].set_title(f'Completion Rate (MA-{window})')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Abandonment rate
    abandonment_ma = np.convolve(training_history['abandonments'], 
                                 np.ones(window)/window, mode='valid')
    axes[1, 1].plot(abandonment_ma)
    axes[1, 1].set_title(f'Abandonment Rate (MA-{window})')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    print(f"✓ Saved training curves to {output_dir / 'training_curves.png'}")
    plt.close()


def main():
    """Main training function."""
    print("\n" + "=" * 70)
    print("RL Agent Training for Adaptive UX")
    print("=" * 70)
    
    # Load personas
    personas = load_personas()
    if personas is None:
        return
    
    # Initialize environment
    env = initialize_environment(personas)
    
    # Initialize agent
    agent = initialize_agent()
    
    # Train agent
    training_history = train_agent(agent, env, num_episodes=100)
    
    # Evaluate agent
    eval_results = evaluate_agent(agent, env, num_episodes=20)
    
    # Save results
    output_dir = save_results(agent, training_history, eval_results)
    
    # Plot results
    plot_training_curves(training_history, output_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\n✅ Successfully trained RL agent!")
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"\n📊 Final Performance:")
    print(f"   • Completion Rate: {eval_results['completion_rate']:.1%}")
    print(f"   • Abandonment Rate: {eval_results['abandonment_rate']:.1%}")
    print(f"   • Avg Cognitive Load: {eval_results['avg_cognitive_load']:.2f}")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review training curves: {output_dir}/training_curves.png")
    print(f"   2. Test agent with different personas")
    print(f"   3. Deploy for real-time adaptation")


if __name__ == "__main__":
    main()
