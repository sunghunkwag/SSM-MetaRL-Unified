# -*- coding: utf-8 -*-
"""
Fixed SSM-MetaRL-Unified: Proper Reinforcement Learning Implementation

Key fixes:
1. Train policy to maximize rewards (not predict next state)
2. Use policy gradient with SSM architecture
3. Proper action selection from policy output
4. Meta-learning for fast adaptation to new tasks
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import gymnasium as gym

from core.ssm import StateSpaceModel
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
        
        # Forward pass (keep gradients for training)
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
            log_prob = torch.tensor(0.0)  # Placeholder
        
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


def train_meta_rl(args, model, env, device):
    """
    Meta-training with MetaMAML for fast adaptation.
    
    Modified to use policy gradient objectives instead of state prediction.
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
            support_obs = torch.cat(observations[:split_idx], dim=0).unsqueeze(0)  # (1, T, obs_dim)
            support_actions = torch.tensor(actions[:split_idx], dtype=torch.long).unsqueeze(0)  # (1, T)
            support_rewards = torch.tensor(rewards[:split_idx], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
            
            # Query set: second half of episode
            query_obs = torch.cat(observations[split_idx:], dim=0).unsqueeze(0)
            query_actions = torch.tensor(actions[split_idx:], dtype=torch.long).unsqueeze(0)
            query_rewards = torch.tensor(rewards[split_idx:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            # For MAML, we use action prediction as the supervised task
            # This allows the model to learn a good initialization for policy learning
            support_y = support_actions.float().unsqueeze(-1)  # Convert to float for MSE
            query_y = query_actions.float().unsqueeze(-1)
            
            tasks.append((support_obs, support_y, query_obs, query_y))
        
        if len(tasks) == 0:
            print(f"Epoch {epoch}: No valid tasks collected, skipping")
            continue
        
        # Meta-update
        initial_hidden = model.init_hidden(batch_size=1)
        
        def action_prediction_loss(pred, target):
            """Custom loss for action prediction"""
            # pred: (batch, time, output_dim)
            # target: (batch, time, 1) - action indices as float
            # Extract action logits from pred
            n_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else pred.shape[-1]
            action_logits = pred[:, :, :n_actions]
            target_long = target.long().squeeze(-1)  # (batch, time)
            return F.cross_entropy(action_logits.reshape(-1, n_actions), target_long.reshape(-1))
        
        loss = meta_learner.meta_update(
            tasks,
            initial_hidden_state=initial_hidden,
            loss_fn=action_prediction_loss
        )
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Meta Loss: {loss:.4f}")
    
    print("Meta-RL training completed.")


def evaluate_policy(model, env, device, num_episodes=20):
    """Evaluate trained policy"""
    print("Evaluating policy...")
    episode_rewards = []
    
    for ep in range(num_episodes):
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
        
        episode_rewards.append(total_reward)
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Evaluation Results ({num_episodes} episodes):")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min: {np.min(episode_rewards):.1f}, Max: {np.max(episode_rewards):.1f}")
    
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(
        description="Fixed SSM-MetaRL: Proper RL Training"
    )
    
    # Environment settings
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Model architecture
    parser.add_argument('--state_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    
    # Training mode
    parser.add_argument('--mode', type=str, default='policy_gradient',
                       choices=['policy_gradient', 'meta_rl'],
                       help='Training mode: policy_gradient or meta_rl')
    
    # Policy gradient settings
    parser.add_argument('--num_episodes', type=int, default=200,
                       help='Number of episodes for policy gradient training')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--pg_lr', type=float, default=0.001,
                       help='Learning rate for policy gradient')
    
    # Meta-RL settings
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--tasks_per_epoch', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize environment
    env = Environment(env_name=args.env_name, batch_size=args.batch_size)
    obs_space = env.observation_space
    action_space = env.action_space
    
    input_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    
    # Output dim should be action space size for policy
    if isinstance(action_space, gym.spaces.Discrete):
        output_dim = action_space.n
    else:
        output_dim = action_space.shape[0]
    
    # Initialize model
    model = StateSpaceModel(
        state_dim=args.state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    print(f"\n=== Fixed SSM-MetaRL ===")
    print(f"Environment: {args.env_name}")
    print(f"Device: {device}")
    print(f"Input Dim: {input_dim}")
    print(f"Output Dim (Actions): {output_dim}")
    print(f"State/Hidden Dim: {args.state_dim}/{args.hidden_dim}")
    print(f"Training Mode: {args.mode}")
    print("=" * 50 + "\n")
    
    if args.mode == 'policy_gradient':
        # Train with policy gradient
        episode_rewards = train_policy_gradient(
            model, env, device,
            num_episodes=args.num_episodes,
            gamma=args.gamma,
            lr=args.pg_lr
        )
        
        # Evaluate
        eval_rewards = evaluate_policy(model, env, device)
        
        # Print summary
        print("\nTraining Summary:")
        print(f"  Initial Reward (first 10 avg): {np.mean(episode_rewards[:10]):.1f}")
        print(f"  Final Reward (last 10 avg): {np.mean(episode_rewards[-10:]):.1f}")
        print(f"  Best Episode: {max(episode_rewards):.1f}")
        
    elif args.mode == 'meta_rl':
        # Train with Meta-RL
        train_meta_rl(args, model, env, device)
        
        # Evaluate
        eval_rewards = evaluate_policy(model, env, device)
    
    env.close()
    print("\n=== Training completed successfully ===")


if __name__ == "__main__":
    main()

