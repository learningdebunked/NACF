#!/usr/bin/env python3
"""Improved RL training with better exploration and reward shaping."""

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
        print(f"❌ Personas not found. Run: python3 generate_personas.py")
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


def train_with_exploration(agent, env, num_episodes=300, exploration_decay=0.995):
    """Train with epsilon-greedy exploration."""
    print("\n" + "=" * 70)
    print(f"Training with Improved Exploration ({num_episodes} episodes)")
    print("=" * 70)
    
    episode_rewards = []
    episode_steps = []
    episode_completions = []
    episode_abandonments = []
    episode_cognitive_loads = []
    action_counts = {i: [] for i in range(5)}
    
    epsilon = 1.0  # Start with high exploration
    min_epsilon = 0.1
    
    print("\nTraining with epsilon-greedy exploration...")
    print("(Agent will explore different actions)\n")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_actions = {i: 0 for i in range(5)}
        cognitive_loads = []
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for step in range(env.max_steps):
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Explore: random action
                action = np.random.randint(0, 5)
                log_prob = 0
            else:
                # Exploit: use policy
                action, log_prob = agent.select_action(state, training=True)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            episode_reward += reward
            episode_actions[action] += 1
            cognitive_loads.append(info.get('cognitive_load', 0))
            
            state = next_state
            
            if done:
                break
        
        # Update agent
        if len(states) > 0:
            agent.update(states, actions, rewards, next_states, dones)
        
        # Decay exploration
        epsilon = max(min_epsilon, epsilon * exploration_decay)
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(len(states))
        episode_completions.append(1 if info.get('completed', False) else 0)
        episode_abandonments.append(1 if info.get('abandoned', False) else 0)
        episode_cognitive_loads.append(np.mean(cognitive_loads) if cognitive_loads else 0)
        
        # Record action diversity
        for action_id in range(5):
            action_counts[action_id].append(episode_actions[action_id])
        
        # Print progress
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_steps = np.mean(episode_steps[-20:])
            completion_rate = np.mean(episode_completions[-20:])
            abandonment_rate = np.mean(episode_abandonments[-20:])
            avg_load = np.mean(episode_cognitive_loads[-20:])
            
            # Action diversity
            recent_actions = {i: np.sum(action_counts[i][-20:]) for i in range(5)}
            total_actions = sum(recent_actions.values())
            action_dist = {i: recent_actions[i]/total_actions if total_actions > 0 else 0 
                          for i in range(5)}
            
            print(f"Episode {episode+1:3d}/{num_episodes} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Steps: {avg_steps:4.1f} | "
                  f"Complete: {completion_rate:.1%} | "
                  f"Abandon: {abandonment_rate:.1%} | "
                  f"Load: {avg_load:.2f} | "
                  f"ε: {epsilon:.2f}")
            
            # Show action diversity every 60 episodes
            if (episode + 1) % 60 == 0:
                print(f"  Action diversity: ", end="")
                for i in range(5):
                    print(f"A{i}:{action_dist[i]:.1%} ", end="")
                print()
    
    print("\n✓ Training complete!")
    
    return {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'completions': episode_completions,
        'abandonments': episode_abandonments,
        'cognitive_loads': episode_cognitive_loads,
        'action_counts': action_counts
    }


def evaluate_agent(agent, env, num_episodes=30):
    """Evaluate the trained agent."""
    print("\n" + "=" * 70)
    print(f"Evaluating Agent ({num_episodes} episodes)")
    print("=" * 70)
    
    eval_rewards = []
    eval_completions = []
    eval_abandonments = []
    eval_cognitive_loads = []
    eval_actions = {i: 0 for i in range(5)}
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        cognitive_loads = []
        
        for step in range(env.max_steps):
            action, _ = agent.select_action(state, training=False)
            eval_actions[action] += 1
            
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
    
    # Action distribution
    total_actions = sum(eval_actions.values())
    action_dist = {i: eval_actions[i]/total_actions if total_actions > 0 else 0 
                   for i in range(5)}
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Completion Rate: {completion_rate:.1%}")
    print(f"  Abandonment Rate: {abandonment_rate:.1%}")
    print(f"  Avg Cognitive Load: {avg_cognitive_load:.2f}")
    
    print(f"\n  Action Distribution:")
    action_names = ["Reduce anim", "Whitespace", "Simplify", "Timeout", "Slow mode"]
    for i in range(5):
        print(f"    {action_names[i]}: {action_dist[i]:.1%}")
    
    return {
        'avg_reward': avg_reward,
        'completion_rate': completion_rate,
        'abandonment_rate': abandonment_rate,
        'avg_cognitive_load': avg_cognitive_load,
        'action_distribution': action_dist
    }


def save_results(agent, training_history, eval_results):
    """Save training results."""
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    
    output_dir = Path('results/rl_improved')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save agent
    agent.save(output_dir / 'rl_agent_improved.pth')
    print(f"✓ Saved agent to {output_dir / 'rl_agent_improved.pth'}")
    
    # Save training history
    with open(output_dir / 'training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)
    print(f"✓ Saved training history")
    
    # Save evaluation results
    with open(output_dir / 'eval_results.pkl', 'wb') as f:
        pickle.dump(eval_results, f)
    print(f"✓ Saved evaluation results")
    
    return output_dir


def plot_improved_results(training_history, eval_results, output_dir):
    """Plot improved training results."""
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Rewards
    axes[0, 0].plot(training_history['rewards'], alpha=0.3)
    window = 20
    rewards_ma = np.convolve(training_history['rewards'], 
                            np.ones(window)/window, mode='valid')
    axes[0, 0].plot(rewards_ma, linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Completion rate
    completion_ma = np.convolve(training_history['completions'], 
                               np.ones(window)/window, mode='valid')
    axes[0, 1].plot(completion_ma, linewidth=2, color='green')
    axes[0, 1].set_title('Completion Rate')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Rate')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cognitive load
    load_ma = np.convolve(training_history['cognitive_loads'], 
                         np.ones(window)/window, mode='valid')
    axes[0, 2].plot(load_ma, linewidth=2, color='orange')
    axes[0, 2].set_title('Cognitive Load')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Load')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].grid(True, alpha=0.3)
    
    # Action diversity over time
    for action_id in range(5):
        action_ma = np.convolve(training_history['action_counts'][action_id], 
                               np.ones(window)/window, mode='valid')
        axes[1, 0].plot(action_ma, label=f'Action {action_id}', linewidth=2)
    axes[1, 0].set_title('Action Usage Over Time')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final action distribution
    action_names = ["Reduce\nanim", "White\nspace", "Simplify", "Timeout", "Slow\nmode"]
    action_values = [eval_results['action_distribution'][i] for i in range(5)]
    axes[1, 1].bar(action_names, action_values, color='steelblue')
    axes[1, 1].set_title('Final Action Distribution')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Performance summary
    metrics = ['Completion\nRate', 'Abandonment\nRate', 'Cognitive\nLoad']
    values = [eval_results['completion_rate'], 
              eval_results['abandonment_rate'],
              eval_results['avg_cognitive_load']]
    colors = ['green', 'red', 'orange']
    axes[1, 2].bar(metrics, values, color=colors)
    axes[1, 2].set_title('Final Performance')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Improved RL Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'improved_training_results.png', dpi=150)
    print(f"✓ Saved to {output_dir / 'improved_training_results.png'}")
    plt.close()


def main():
    """Main training function."""
    print("\n" + "=" * 70)
    print("Improved RL Training with Better Exploration")
    print("=" * 70)
    
    # Load personas
    personas = load_personas()
    if personas is None:
        return
    
    # Initialize environment
    print("\n" + "=" * 70)
    print("Initializing Environment")
    print("=" * 70)
    env = CheckoutEnvironment(personas)
    print(f"✓ Environment ready")
    
    # Initialize agent
    print("\n" + "=" * 70)
    print("Initializing Agent")
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
    
    # Train with exploration
    training_history = train_with_exploration(agent, env, num_episodes=300)
    
    # Evaluate
    eval_results = evaluate_agent(agent, env, num_episodes=30)
    
    # Save results
    output_dir = save_results(agent, training_history, eval_results)
    
    # Plot results
    plot_improved_results(training_history, eval_results, output_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\n✅ Improved training complete!")
    print(f"\n📊 Final Performance:")
    print(f"   • Completion Rate: {eval_results['completion_rate']:.1%}")
    print(f"   • Abandonment Rate: {eval_results['abandonment_rate']:.1%}")
    print(f"   • Cognitive Load: {eval_results['avg_cognitive_load']:.2f}")
    print(f"\n📁 Results: {output_dir}")
    print(f"\n🎯 Next: python3 analyze_rl_results.py")


if __name__ == "__main__":
    main()
