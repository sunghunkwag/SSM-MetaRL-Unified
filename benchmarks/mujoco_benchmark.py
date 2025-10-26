#!/usr/bin/env python3
"""
MuJoCo Benchmark Suite for SSM-MetaRL-Unified

Comprehensive benchmarks for continuous control tasks including:
- HalfCheetah-v5
- Ant-v5
- Hopper-v5
- Walker2d-v5

This addresses the critical gap in the original repository which only
provided CartPole results despite claiming to be a "serious benchmark".

Usage:
    python benchmarks/mujoco_benchmark.py --env HalfCheetah-v5
    python benchmarks/mujoco_benchmark.py --env Ant-v5
    python benchmarks/mujoco_benchmark.py --all
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from datetime import datetime
import gymnasium as gym

from core.ssm import StateSpaceModel
from adaptation.standard_adapter import StandardAdapter, StandardAdaptationConfig
from adaptation.hybrid_adapter import HybridAdapter, HybridAdaptationConfig
from experience.experience_buffer import ExperienceBuffer
from env_runner.environment import Environment


class MuJoCoBenchmark:
    """Benchmark suite for MuJoCo continuous control environments"""
    
    def __init__(self, env_name='HalfCheetah-v5'):
        self.env_name = env_name
        self.results_dir = Path('benchmarks/results/mujoco')
        self.plots_dir = self.results_dir / 'plots'
        self.tables_dir = self.results_dir / 'tables'
        self.models_dir = Path('models/mujoco')
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        self.env = gym.make(env_name)
        
        # Get environment specs
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        print(f"Environment: {env_name}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
        
        # Results storage
        self.results = {
            'standard': [],
            'hybrid': [],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'environment': env_name,
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim
            }
        }
    
    def create_model(self):
        """Create SSM model for continuous control"""
        model = StateSpaceModel(
            state_dim=64,  # Larger state for complex dynamics
            input_dim=self.obs_dim,
            output_dim=self.action_dim * 2,  # Mean and log_std for each action
            hidden_dim=128  # Larger hidden for MuJoCo
        )
        return model
    
    def train_policy(self, num_episodes=500, learning_rate=3e-4, gamma=0.99):
        """
        Train policy using policy gradient for continuous control.
        This is a simplified training for demonstration purposes.
        """
        print(f"\n{'='*60}")
        print(f"Training Policy on {self.env_name}")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Gamma: {gamma}\n")
        
        model = self.create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        episode_rewards = []
        best_reward = -np.inf
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            hidden = model.init_hidden(batch_size=1)
            
            observations = []
            actions = []
            rewards = []
            log_probs = []
            
            done = False
            truncated = False
            steps = 0
            max_steps = 1000
            
            while not (done or truncated) and steps < max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Forward pass
                output, hidden = model(obs_tensor, hidden)
                
                # Split output into mean and log_std
                mean = output[:, :self.action_dim]
                log_std = output[:, self.action_dim:]
                std = torch.exp(log_std.clamp(-20, 2))  # Clamp for stability
                
                # Sample action from Gaussian
                dist = torch.distributions.Normal(mean, std)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor).sum(dim=-1)
                
                # Clip action to valid range
                action = action_tensor.detach().numpy().flatten()
                action = np.clip(action, -1.0, 1.0)
                
                # Step environment
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                observations.append(obs_tensor)
                actions.append(action_tensor)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                obs = next_obs
                steps += 1
            
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)
            
            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns)
            
            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Compute policy loss
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            loss = torch.stack(policy_loss).sum()
            
            # Update policy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track best model
            if total_reward > best_reward:
                best_reward = total_reward
                model_path = self.models_dir / f'{self.env_name.lower().replace("-", "_")}_best.pth'
                model.save(str(model_path))
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"Reward = {total_reward:.1f}, "
                      f"Avg(50) = {avg_reward:.1f}, "
                      f"Best = {best_reward:.1f}")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Reward: {best_reward:.2f}")
        print(f"Final Avg(50): {np.mean(episode_rewards[-50:]):.2f}")
        print(f"{'='*60}\n")
        
        # Save training curve
        self.save_training_curve(episode_rewards)
        
        return model, episode_rewards
    
    def save_training_curve(self, episode_rewards):
        """Save training curve plot"""
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, alpha=0.3, color='blue')
        
        # Plot moving average
        window = 50
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, 
                                    np.ones(window)/window, 
                                    mode='valid')
            plt.plot(range(window-1, len(episode_rewards)), 
                    moving_avg, 
                    color='red', 
                    linewidth=2,
                    label=f'Moving Avg ({window})')
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Training Progress on {self.env_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.plots_dir / f'{self.env_name.lower().replace("-", "_")}_training.png'
        plt.savefig(plot_path, dpi=300)
        print(f"✓ Training curve saved: {plot_path}")
        plt.close()
    
    def evaluate_model(self, model, num_episodes=20, mode='standard'):
        """Evaluate trained model"""
        print(f"\n{'='*60}")
        print(f"Evaluating {mode.upper()} mode on {self.env_name}")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes}\n")
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            hidden = model.init_hidden(batch_size=1)
            total_reward = 0
            done = False
            truncated = False
            steps = 0
            max_steps = 1000
            
            while not (done or truncated) and steps < max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                with torch.no_grad():
                    output, hidden = model(obs_tensor, hidden)
                
                # Use mean action (no sampling during evaluation)
                mean = output[:, :self.action_dim]
                action = mean.numpy().flatten()
                action = np.clip(action, -1.0, 1.0)
                
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if (ep + 1) % 5 == 0:
                print(f"Episode {ep+1}/{num_episodes}: "
                      f"Reward = {total_reward:.1f}, "
                      f"Steps = {steps}")
        
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results:")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Min: {min(episode_rewards):.1f}, Max: {max(episode_rewards):.1f}")
        print(f"  Average Episode Length: {avg_length:.1f}")
        print(f"{'='*60}\n")
        
        return {
            'mode': mode,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'min_reward': min(episode_rewards),
            'max_reward': max(episode_rewards),
            'avg_length': avg_length,
            'all_rewards': episode_rewards,
            'all_lengths': episode_lengths
        }
    
    def run_full_benchmark(self, train_episodes=500):
        """Run complete benchmark: train and evaluate"""
        print("\n" + "="*60)
        print(f"MuJoCo Benchmark Suite: {self.env_name}")
        print("="*60)
        
        # Train policy
        model, training_rewards = self.train_policy(num_episodes=train_episodes)
        
        # Load best model
        model_path = self.models_dir / f'{self.env_name.lower().replace("-", "_")}_best.pth'
        if model_path.exists():
            model.load(str(model_path))
            print(f"✓ Loaded best model from {model_path}")
        
        # Evaluate
        result = self.evaluate_model(model, num_episodes=20, mode='trained')
        self.results['standard'].append(result)
        
        # Save results
        self.save_results()
        
        # Generate comparison table
        self.generate_tables()
        
        print("\n" + "="*60)
        print("Benchmark Complete!")
        print("="*60)
        print(f"Results saved to: {self.results_dir}")
        print(f"Plots saved to: {self.plots_dir}")
        print(f"Tables saved to: {self.tables_dir}")
        print(f"Model saved to: {model_path}")
        
        return result
    
    def save_results(self):
        """Save results to JSON"""
        output_file = self.results_dir / f'{self.env_name.lower().replace("-", "_")}_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to {output_file}")
    
    def generate_tables(self):
        """Generate comparison tables"""
        md_content = f"# {self.env_name} Benchmark Results\n\n"
        md_content += f"**Date**: {self.results['metadata']['timestamp']}\n\n"
        md_content += f"**Environment**: {self.results['metadata']['environment']}\n\n"
        md_content += f"**Observation Dimension**: {self.results['metadata']['obs_dim']}\n\n"
        md_content += f"**Action Dimension**: {self.results['metadata']['action_dim']}\n\n"
        
        md_content += "## Evaluation Results\n\n"
        md_content += "| Mode | Avg Reward | Std | Min | Max | Avg Length |\n"
        md_content += "|------|------------|-----|-----|-----|------------|\n"
        
        for r in self.results['standard']:
            md_content += f"| {r['mode']} | {r['avg_reward']:.2f} | {r['std_reward']:.2f} | "
            md_content += f"{r['min_reward']:.1f} | {r['max_reward']:.1f} | {r['avg_length']:.1f} |\n"
        
        # Save markdown table
        table_file = self.tables_dir / f'{self.env_name.lower().replace("-", "_")}_results.md'
        with open(table_file, 'w') as f:
            f.write(md_content)
        print(f"✓ Table saved: {table_file}")


def run_all_benchmarks(train_episodes=500):
    """Run benchmarks on all MuJoCo environments"""
    environments = [
        'HalfCheetah-v5',
        'Ant-v5',
        'Hopper-v5',
        'Walker2d-v5'
    ]
    
    all_results = {}
    
    for env_name in environments:
        print(f"\n\n{'#'*60}")
        print(f"# Starting benchmark for {env_name}")
        print(f"{'#'*60}\n")
        
        try:
            benchmark = MuJoCoBenchmark(env_name=env_name)
            result = benchmark.run_full_benchmark(train_episodes=train_episodes)
            all_results[env_name] = result
        except Exception as e:
            print(f"ERROR: Failed to run benchmark for {env_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate summary table
    generate_summary_table(all_results)
    
    return all_results


def generate_summary_table(all_results):
    """Generate summary table comparing all environments"""
    results_dir = Path('benchmarks/results/mujoco')
    tables_dir = results_dir / 'tables'
    
    md_content = "# MuJoCo Benchmark Summary\n\n"
    md_content += f"**Date**: {datetime.now().isoformat()}\n\n"
    md_content += "## Performance Across Environments\n\n"
    md_content += "| Environment | Avg Reward | Std | Min | Max | Avg Length |\n"
    md_content += "|-------------|------------|-----|-----|-----|------------|\n"
    
    for env_name, result in sorted(all_results.items()):
        md_content += f"| {env_name} | {result['avg_reward']:.2f} | {result['std_reward']:.2f} | "
        md_content += f"{result['min_reward']:.1f} | {result['max_reward']:.1f} | {result['avg_length']:.1f} |\n"
    
    md_content += "\n## Environment Details\n\n"
    md_content += "| Environment | Observation Dim | Action Dim | Description |\n"
    md_content += "|-------------|-----------------|------------|-------------|\n"
    md_content += "| HalfCheetah-v5 | 17 | 6 | 2D robot, learns to run forward |\n"
    md_content += "| Ant-v5 | 27 | 8 | 3D quadruped, learns to walk |\n"
    md_content += "| Hopper-v5 | 11 | 3 | 2D monoped, learns to hop |\n"
    md_content += "| Walker2d-v5 | 17 | 6 | 2D biped, learns to walk |\n"
    
    # Save summary table
    summary_file = tables_dir / 'mujoco_summary.md'
    with open(summary_file, 'w') as f:
        f.write(md_content)
    print(f"\n✓ Summary table saved: {summary_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MuJoCo Benchmark Suite for SSM-MetaRL-Unified"
    )
    parser.add_argument('--env', type=str, default='HalfCheetah-v5',
                       help='Environment name (HalfCheetah-v5, Ant-v5, Hopper-v5, Walker2d-v5)')
    parser.add_argument('--all', action='store_true',
                       help='Run benchmarks on all environments')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_benchmarks(train_episodes=args.episodes)
    else:
        benchmark = MuJoCoBenchmark(env_name=args.env)
        benchmark.run_full_benchmark(train_episodes=args.episodes)

