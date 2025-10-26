#!/usr/bin/env python3
"""
Improved Training Script for SSM-MetaRL-Unified

Trains a better-performing model with optimized hyperparameters.

Usage:
    python train_improved_model.py --epochs 100 --mode hybrid
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from env_runner.environment import Environment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_improved_model(
    num_epochs=100,
    tasks_per_epoch=10,
    state_dim=64,  # Increased from 32
    hidden_dim=128,  # Increased from 64
    inner_lr=0.01,
    outer_lr=0.001,
    adaptation_steps=10,  # Increased from 5
    mode='hybrid'
):
    """Train an improved model with better hyperparameters"""
    
    logger.info("="*60)
    logger.info("Training Improved SSM-MetaRL Model")
    logger.info("="*60)
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Tasks per epoch: {tasks_per_epoch}")
    logger.info(f"State dim: {state_dim}")
    logger.info(f"Hidden dim: {hidden_dim}")
    logger.info(f"Inner LR: {inner_lr}")
    logger.info(f"Outer LR: {outer_lr}")
    logger.info(f"Adaptation steps: {adaptation_steps}")
    logger.info(f"Mode: {mode}")
    logger.info("="*60)
    
    # Create environment
    env = Environment('CartPole-v1')
    
    # Create model with larger capacity
    model = StateSpaceModel(
        state_dim=state_dim,
        input_dim=4,  # CartPole observation space
        output_dim=4,  # Output dimension for SSM
        hidden_dim=hidden_dim
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create meta-learner
    meta_learner = MetaMAML(
        model=model,
        inner_lr=inner_lr,
        outer_lr=outer_lr
    )
    
    logger.info("Starting meta-training...")
    
    # Training loop
    best_reward = 0
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_rewards = []
        
        for task_idx in range(tasks_per_epoch):
            # Generate task data
            support_obs, support_actions, query_obs, query_actions = generate_task_data(
                env, model, num_episodes=3
            )
            
            # Meta-training step
            loss = meta_learner.meta_train_step(
                support_x=support_obs,
                support_y=support_actions,
                query_x=query_obs,
                query_y=query_actions
            )
            
            epoch_losses.append(loss)
            
            # Evaluate on a test episode
            reward = evaluate_model(env, model)
            epoch_rewards.append(reward)
        
        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean(epoch_rewards)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Reward = {avg_reward:.2f}")
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            model.save(f'models/cartpole_{mode}_improved_model.pth')
            logger.info(f"  → New best model saved! Reward: {avg_reward:.2f}")
    
    logger.info("="*60)
    logger.info(f"Training complete! Best reward: {best_reward:.2f}")
    logger.info(f"Model saved to: models/cartpole_{mode}_improved_model.pth")
    logger.info("="*60)
    
    return model


def generate_task_data(env, model, num_episodes=3):
    """Generate training data from environment episodes"""
    all_obs = []
    all_actions = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        hidden = model.init_hidden(batch_size=1)
        
        episode_obs = []
        episode_actions = []
        
        for step in range(200):  # Max 200 steps per episode
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, hidden = model(obs_tensor, hidden)
            
            # Use first 2 dimensions for action (CartPole has 2 actions)
            action = torch.argmax(action_logits[:, :2], dim=-1).item()
            
            episode_obs.append(obs)
            episode_actions.append(action)
            
            next_obs, reward, done, info = env.step(action)
            
            if done:
                break
            
            obs = next_obs
        
        all_obs.extend(episode_obs)
        all_actions.extend(episode_actions)
    
    # Convert to tensors
    obs_tensor = torch.FloatTensor(all_obs).unsqueeze(0)  # (1, T, 4)
    actions_tensor = torch.LongTensor(all_actions).unsqueeze(0).unsqueeze(-1).float()  # (1, T, 1)
    
    # Split into support and query sets
    split_idx = len(all_obs) // 2
    support_obs = obs_tensor[:, :split_idx, :]
    support_actions = actions_tensor[:, :split_idx, :]
    query_obs = obs_tensor[:, split_idx:, :]
    query_actions = actions_tensor[:, split_idx:, :]
    
    return support_obs, support_actions, query_obs, query_actions


def evaluate_model(env, model, num_episodes=5):
    """Evaluate model performance"""
    total_rewards = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        hidden = model.init_hidden(batch_size=1)
        episode_reward = 0
        
        for step in range(500):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, hidden = model(obs_tensor, hidden)
            
            # Use first 2 dimensions for action
            action = torch.argmax(action_logits[:, :2], dim=-1).item()
            
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                break
            
            obs = next_obs
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train improved SSM-MetaRL model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--tasks_per_epoch', type=int, default=10, help='Tasks per epoch')
    parser.add_argument('--state_dim', type=int, default=64, help='State dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Outer loop learning rate')
    parser.add_argument('--adaptation_steps', type=int, default=10, help='Adaptation steps')
    parser.add_argument('--mode', type=str, default='hybrid', choices=['standard', 'hybrid'],
                       help='Adaptation mode')
    
    args = parser.parse_args()
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Train model
    model = train_improved_model(
        num_epochs=args.epochs,
        tasks_per_epoch=args.tasks_per_epoch,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        adaptation_steps=args.adaptation_steps,
        mode=args.mode
    )
    
    # Final evaluation
    env = Environment('CartPole-v1')
    final_reward = evaluate_model(env, model, num_episodes=20)
    logger.info(f"\nFinal evaluation (20 episodes): {final_reward:.2f} ± {np.std([evaluate_model(env, model, 1) for _ in range(20)]):.2f}")
    
    # Save training log
    with open(f'logs/training_improved_{args.mode}.log', 'w') as f:
        f.write(f"Training completed\n")
        f.write(f"Final reward: {final_reward:.2f}\n")
        f.write(f"Model: models/cartpole_{args.mode}_improved_model.pth\n")

