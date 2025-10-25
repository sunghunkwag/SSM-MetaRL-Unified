#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script to create actual model weights (.pth file)
Based on the guide specifications for SSM-MetaRL-Unified
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import OrderedDict

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation import HybridAdapter, HybridAdaptationConfig
from experience.experience_buffer import ExperienceBuffer
from env_runner.environment import Environment


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns"""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


def collect_episode(env, policy_model, device='cpu', max_steps=200, experience_buffer=None):
    """Collect a single episode using the SSM policy"""
    observations = []
    actions = []
    rewards = []
    log_probs = []
    
    obs = env.reset()
    hidden_state = policy_model.init_hidden(batch_size=1)
    
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Forward pass through SSM
        action_logits, next_hidden_state = policy_model(obs_tensor, hidden_state)
        
        # Get action probabilities
        if isinstance(env.action_space, gym.spaces.Discrete):
            n_actions = env.action_space.n if hasattr(env.action_space, 'n') else 2
            logits = action_logits[:, :n_actions]
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)
        else:
            action = action_logits.cpu().numpy().flatten()
            log_prob = torch.tensor(0.0)
        
        next_obs, reward, done, info = env.step(action)
        
        # Add to experience buffer if provided
        if experience_buffer is not None:
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            experience_buffer.add(obs_tensor, next_obs_tensor)
        
        observations.append(obs_tensor)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        
        obs = next_obs
        hidden_state = next_hidden_state
        steps += 1
    
    return observations, actions, rewards, log_probs


def train_meta(args, model, env, device, experience_buffer=None):
    """
    Meta-training with MetaMAML
    """
    print("\n=== Starting Meta-RL Training with MAML ===")
    print(f"Environment: {args.env_name}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Tasks per epoch: {args.tasks_per_epoch}")
    print(f"Adaptation mode: {args.adaptation_mode}")
    print("=" * 60)
    
    meta_learner = MetaMAML(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr
    )
    
    n_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    epoch_rewards = []
    
    for epoch in range(args.num_epochs):
        # Collect multiple tasks
        tasks = []
        task_rewards = []
        
        for task_idx in range(args.tasks_per_epoch):
            # Collect episode
            observations, actions, rewards, log_probs = collect_episode(
                env, model, device, max_steps=100, experience_buffer=experience_buffer
            )
            
            task_rewards.append(sum(rewards))
            
            if len(observations) < 10:
                continue
            
            # Split into support and query sets
            split_idx = len(observations) // 2
            
            # Support set
            support_obs = torch.cat(observations[:split_idx], dim=0).unsqueeze(0)
            support_actions = torch.tensor(actions[:split_idx], dtype=torch.long).unsqueeze(0)
            
            # Query set
            query_obs = torch.cat(observations[split_idx:], dim=0).unsqueeze(0)
            query_actions = torch.tensor(actions[split_idx:], dtype=torch.long).unsqueeze(0)
            
            # Use action prediction as supervised task for MAML
            support_y = support_actions.float().unsqueeze(-1)
            query_y = query_actions.float().unsqueeze(-1)
            
            tasks.append((support_obs, support_y, query_obs, query_y))
        
        if len(tasks) == 0:
            print(f"Epoch {epoch}: No valid tasks collected, skipping")
            continue
        
        # Meta-update
        initial_hidden = model.init_hidden(batch_size=1)
        
        def action_prediction_loss(pred, target):
            action_logits = pred[:, :, :n_actions]
            target_long = target.long().squeeze(-1)
            return F.cross_entropy(action_logits.reshape(-1, n_actions), target_long.reshape(-1))
        
        meta_loss = meta_learner.meta_update(
            tasks,
            initial_hidden_state=initial_hidden,
            loss_fn=action_prediction_loss
        )
        
        avg_reward = np.mean(task_rewards) if task_rewards else 0
        epoch_rewards.append(avg_reward)
        
        if epoch % 10 == 0:
            recent_avg = np.mean(epoch_rewards[-10:]) if len(epoch_rewards) >= 10 else avg_reward
            print(f"Epoch {epoch:4d}: Meta-Loss={meta_loss:8.4f}, Avg Reward={avg_reward:6.1f}, Recent={recent_avg:6.1f}, Buffer={len(experience_buffer) if experience_buffer else 0}")
    
    print("\n" + "=" * 60)
    print("Meta-training completed!")
    print(f"Initial Avg Reward: {np.mean(epoch_rewards[:10]) if len(epoch_rewards) >= 10 else np.mean(epoch_rewards):6.1f}")
    print(f"Final Avg Reward:   {np.mean(epoch_rewards[-10:]) if len(epoch_rewards) >= 10 else np.mean(epoch_rewards):6.1f}")
    print(f"Best Epoch:         {max(epoch_rewards) if epoch_rewards else 0:6.1f}")
    print("=" * 60 + "\n")


def test_time_adapt(args, model, env, device, adaptation_mode='hybrid', experience_buffer=None):
    """
    Test-time adaptation (optional - just for demonstration)
    """
    print(f"\n=== Test-Time Adaptation ({adaptation_mode.upper()}) ===")
    print("Running a few test episodes to verify model performance...")
    
    test_rewards = []
    for ep in range(5):
        obs = env.reset()
        hidden_state = model.init_hidden(batch_size=1)
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_logits, hidden_state = model(obs_tensor, hidden_state)
            
            if isinstance(env.action_space, gym.spaces.Discrete):
                n_actions = env.action_space.n
                logits = action_logits[:, :n_actions]
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = action_logits.cpu().numpy().flatten()
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        test_rewards.append(total_reward)
    
    print(f"Test Results: Avg Reward = {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train and Save SSM-MetaRL Model Weights"
    )
    
    # Environment settings
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Model architecture
    parser.add_argument('--state_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    
    # Meta-RL settings
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of meta-training epochs')
    parser.add_argument('--tasks_per_epoch', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.001)
    
    # Adaptation mode
    parser.add_argument('--adaptation_mode', type=str, default='hybrid',
                       choices=['standard', 'hybrid'],
                       help='Adaptation mode: standard or hybrid')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Initialize environment
    env = Environment(env_name=args.env_name, batch_size=args.batch_size)
    obs_space = env.observation_space
    action_space = env.action_space
    
    input_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    output_dim = input_dim  # For state prediction in adaptation
    
    # Initialize experience buffer
    experience_buffer = ExperienceBuffer(max_size=10000, device=str(device))
    
    # Initialize model
    model = StateSpaceModel(
        state_dim=args.state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    print(f"Model Architecture:")
    print(f"  Input Dim: {input_dim}")
    print(f"  Output Dim: {output_dim}")
    print(f"  State Dim: {args.state_dim}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print()
    
    # Meta-train with MetaMAML
    train_meta(args, model, env, device, experience_buffer=experience_buffer)
    
    # ===== SAVE MODEL WEIGHTS =====
    print(f"\n=== [SUCCESS] Meta-training completed. Saving 'real model' weights... ===")
    model_filename = f"cartpole_{args.adaptation_mode}_real_model.pth"
    model.save(model_filename)
    print(f"=== [SAVED] Filename: {model_filename} ===\n")
    # ===== END SAVE =====
    
    # Test-time adaptation (optional demonstration)
    test_time_adapt(
        args, model, env, device, 
        adaptation_mode=args.adaptation_mode,
        experience_buffer=experience_buffer
    )
    
    env.close()
    print("\n=== Training and model saving completed successfully ===\n")


if __name__ == "__main__":
    main()

