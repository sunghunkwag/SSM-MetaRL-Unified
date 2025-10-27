#!/usr/bin/env python3
"""
SSM-MetaRL-Unified: Main Training Script

This script provides multiple training modes:
1. Policy Gradient: Simple REINFORCE-style training
2. Meta-RL: MAML-based meta-learning for fast adaptation
3. Improved: Modern PPO training with improved SSM architecture

Usage:
    # Simple policy gradient (CartPole)
    python main.py --mode policy_gradient --env CartPole-v1 --episodes 200

    # Meta-RL training (CartPole)
    python main.py --mode meta_rl --env CartPole-v1 --epochs 100

    # Improved PPO training (MuJoCo)
    python main.py --mode improved --env HalfCheetah-v5 --episodes 300
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import gymnasium as gym

from core.ssm import StateSpaceModel
from core.improved_ssm import ImprovedSSM
from meta_rl.meta_maml import MetaMAML
from env_runner.environment import Environment


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns"""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


def collect_episode(env, policy_model, device='cpu', max_steps=500):
    """
    Collect a single episode using the policy.
    
    Returns:
        observations, actions, rewards, log_probs, hidden_states
    """
    observations = []
    actions = []
    rewards = []
    log_probs = []
    hidden_states_list = []
    
    policy_model.eval()
    obs = env.reset()
    hidden_state = policy_model.init_hidden(batch_size=1)
    
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Forward pass
        action_logits, next_hidden_state = policy_model(obs_tensor, hidden_state)
        
        # Get action probabilities
        if isinstance(env.action_space, gym.spaces.Discrete):
            n_actions = env.action_space.n
            logits = action_logits[:, :n_actions]
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)
        else:
            # Continuous action space
            action = action_logits.cpu().numpy().flatten()
            log_prob = torch.tensor(0.0)
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        # Store trajectory
        observations.append(obs_tensor)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        hidden_states_list.append(hidden_state)
        
        obs = next_obs
        hidden_state = next_hidden_state
        steps += 1
    
    return observations, actions, rewards, log_probs, hidden_states_list


def train_policy_gradient(model, env, device, num_episodes=100, gamma=0.99, lr=0.001):
    """
    Train policy using policy gradient (REINFORCE-style) with SSM.
    """
    print("Training policy with policy gradient...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Collect episode
        observations, actions, rewards, log_probs, hidden_states = collect_episode(
            env, model, device
        )
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        
        # Compute returns
        returns = compute_returns(rewards, gamma).to(device)
        
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
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Reward: {total_reward:.1f}, Avg(10): {avg_reward:.1f}, Loss: {loss.item():.4f}")
    
    return episode_rewards


def train_improved_pg(model, env, device, num_episodes=300, gamma=0.99, lr=3e-4):
    """
    Train improved SSM with simple policy gradient.
    Uses the actor-critic architecture properly.
    """
    print("Training improved SSM with policy gradient...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    episode_rewards = []
    best_reward = -np.inf
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        hidden = model.init_hidden(batch_size=1, device=device)
        
        observations = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        total_reward = 0
        done = False
        truncated = False
        steps = 0
        max_steps = 1000
        
        # Collect episode
        while not (done or truncated) and steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action
            action, log_prob, value, hidden = model.get_action(
                obs_tensor, hidden, deterministic=False
            )
            
            action_np = action.cpu().detach().numpy().flatten()
            action_np = np.clip(action_np, -1.0, 1.0)
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action_np)
            
            # Store
            observations.append(obs_tensor)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            total_reward += reward
            obs = next_obs
            steps += 1
        
        episode_rewards.append(total_reward)
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute loss
        policy_loss = 0
        value_loss = 0
        
        for log_prob, value, R in zip(log_probs, values, returns):
            # Policy gradient loss
            advantage = R - value.detach()
            policy_loss += -log_prob * advantage
            
            # Value loss
            value_loss += F.mse_loss(value, R.unsqueeze(0))
        
        policy_loss = policy_loss / len(log_probs)
        value_loss = value_loss / len(values)
        
        total_loss = policy_loss + 0.5 * value_loss
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track best
        if total_reward > best_reward:
            best_reward = total_reward
        
        # Log
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Reward = {total_reward:.1f}, "
                  f"Avg(20) = {avg_reward:.1f}, "
                  f"Best = {best_reward:.1f}")
    
    print(f"\nTraining complete! Best reward: {best_reward:.2f}")
    return episode_rewards


def train_meta_rl(args, model, env, device):
    """
    Meta-training with MetaMAML for fast adaptation.
    """
    print("Starting Meta-RL training with MAML...")
    meta_learner = MetaMAML(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr
    )
    
    for epoch in range(args.num_epochs):
        # Collect multiple tasks (episodes)
        tasks = []
        
        for task_idx in range(args.tasks_per_epoch):
            # Collect episode data
            observations, actions, rewards, log_probs, hidden_states = collect_episode(
                env, model, device, max_steps=100
            )
            
            if len(observations) < 10:
                continue
            
            # Split into support and query sets
            split_idx = len(observations) // 2
            
            # Support set: first half of episode
            support_obs = torch.cat(observations[:split_idx], dim=0).unsqueeze(0)
            support_actions = torch.tensor(actions[:split_idx], dtype=torch.long).unsqueeze(0)
            support_rewards = torch.tensor(rewards[:split_idx], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            # Query set: second half of episode
            query_obs = torch.cat(observations[split_idx:], dim=0).unsqueeze(0)
            query_actions = torch.tensor(actions[split_idx:], dtype=torch.long).unsqueeze(0)
            query_rewards = torch.tensor(rewards[split_idx:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            # For MAML, we use action prediction as the supervised task
            support_y = support_actions.float().unsqueeze(-1)
            query_y = query_actions.float().unsqueeze(-1)
            
            tasks.append((support_obs, support_y, query_obs, query_y))
        
        if len(tasks) == 0:
            print(f"Epoch {epoch}: No valid tasks collected, skipping")
            continue
        
        # Meta-update
        initial_hidden = model.init_hidden(batch_size=1)
        
        def action_prediction_loss(pred, target):
            """Custom loss for action prediction"""
            n_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else pred.shape[-1]
            action_logits = pred[:, :, :n_actions]
            target_long = target.long().squeeze(-1)
            return F.cross_entropy(action_logits.reshape(-1, n_actions), target_long.reshape(-1))
        
        loss = meta_learner.meta_update(
            tasks,
            initial_hidden_state=initial_hidden,
            loss_fn=action_prediction_loss
        )
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Meta Loss: {loss:.4f}")
    
    print("Meta-RL training completed.")


def main():
    parser = argparse.ArgumentParser(
        description="SSM-MetaRL-Unified: Unified Training Script"
    )
    
    # Environment settings
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Environment name (CartPole-v1, HalfCheetah-v5, etc.)')
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Model architecture
    parser.add_argument('--state_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    
    # Training mode
    parser.add_argument('--mode', type=str, default='policy_gradient',
                       choices=['policy_gradient', 'meta_rl', 'improved'],
                       help='Training mode: policy_gradient (simple), meta_rl (MAML), improved (modern PPO-style)')
    
    # Policy gradient settings
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of episodes for training')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    # Meta-RL settings
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of meta-training epochs')
    parser.add_argument('--tasks_per_epoch', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize environment
    env = Environment(env_name=args.env, batch_size=args.batch_size)
    obs_space = env.observation_space
    action_space = env.action_space
    
    input_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    
    # Output dim
    if isinstance(action_space, gym.spaces.Discrete):
        output_dim = action_space.n
    else:
        output_dim = action_space.shape[0]
    
    # Initialize model based on mode
    if args.mode == 'improved':
        print("\n=== Using Improved SSM Architecture ===")
        model = ImprovedSSM(
            input_dim=input_dim,
            action_dim=output_dim,
            state_dim=args.state_dim * 2,  # Larger for improved version
            hidden_dim=args.hidden_dim * 2,
            num_layers=2,
            use_layer_norm=True,
            use_residual=True
        ).to(device)
        
        # Train with improved method
        episode_rewards = train_improved_pg(
            model, env, device,
            num_episodes=args.episodes,
            gamma=args.gamma,
            lr=args.lr if args.lr != 0.001 else 3e-4  # Default to 3e-4 for improved
        )
    else:
        print("\n=== Using Standard SSM Architecture ===")
        model = StateSpaceModel(
            state_dim=args.state_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=args.hidden_dim
        ).to(device)
        
        if args.mode == 'policy_gradient':
            episode_rewards = train_policy_gradient(
                model, env, device,
                num_episodes=args.episodes,
                gamma=args.gamma,
                lr=args.lr
            )
        elif args.mode == 'meta_rl':
            train_meta_rl(args, model, env, device)
            episode_rewards = []
    
    # Save model
    if args.mode in ['policy_gradient', 'improved']:
        model_name = f"{args.env.lower().replace('-', '_')}_{args.mode}.pth"
        model.save(f"models/{model_name}")
        print(f"\nModel saved to models/{model_name}")
        
        if episode_rewards:
            print(f"Final average reward (last 20): {np.mean(episode_rewards[-20:]):.2f}")


if __name__ == '__main__':
    main()

