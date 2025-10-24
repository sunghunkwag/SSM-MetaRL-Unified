"""
Serious Benchmark Suite for SSM-MetaRL-Unified

This script runs comprehensive benchmarks on high-dimensional tasks with
SOTA baseline comparisons and support for experience-augmented adaptation.

Usage:
    python experiments/serious_benchmark.py --task halfcheetah-vel --method ssm --adaptation_mode standard
    python experiments/serious_benchmark.py --task ant-dir --method ssm --adaptation_mode hybrid --epochs 100
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation import StandardAdapter, StandardAdaptationConfig
from adaptation import HybridAdapter, HybridAdaptationConfig
from experience.experience_buffer import ExperienceBuffer
from experiments.task_distributions import get_task_distribution, list_task_distributions
from experiments.baselines import get_baseline_policy, list_baselines
import gymnasium as gym


class BenchmarkRunner:
    """Runs meta-learning benchmarks and collects metrics."""
    
    def __init__(self, task_dist_name: str, method_name: str, config: Dict[str, Any]):
        self.task_dist_name = task_dist_name
        self.method_name = method_name
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Create task distribution
        self.task_dist = get_task_distribution(task_dist_name)
        
        # Get environment info from first task
        sample_env = self.task_dist.sample_task(0)
        self.state_dim = sample_env.observation_space.shape[0]
        self.action_dim = sample_env.action_space.shape[0]
        sample_env.close()
        
        # Create model
        self.model = self._create_model()
        
        # Create meta-learner
        self.meta_learner = MetaMAML(
            model=self.model,
            inner_lr=config.get('inner_lr', 0.01),
            outer_lr=config.get('outer_lr', 0.001)
        )
        
        # Initialize experience buffer if using hybrid mode
        self.experience_buffer = None
        if config.get('adaptation_mode', 'standard') == 'hybrid':
            self.experience_buffer = ExperienceBuffer(
                max_size=config.get('buffer_size', 10000),
                device=str(self.device)
            )
            print(f"Initialized ExperienceBuffer with max_size={config.get('buffer_size', 10000)}")
        
        # Metrics storage
        self.metrics = defaultdict(list)
    
    def _create_model(self) -> nn.Module:
        """Create model based on method name."""
        hidden_dim = self.config.get('hidden_dim', 128)
        
        # For meta-learning, we predict next observation (state prediction task)
        output_dim = self.state_dim
        
        if self.method_name == 'ssm':
            from core.ssm import StateSpaceModel
            return StateSpaceModel(
                state_dim=hidden_dim,
                input_dim=self.state_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim
            ).to(self.device)
        
        elif self.method_name in ['mlp', 'lstm', 'gru', 'transformer']:
            return get_baseline_policy(
                self.method_name,
                self.state_dim,
                output_dim,
                hidden_dim
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown method: {self.method_name}")
    
    def collect_episode_data(
        self, 
        env, 
        max_steps: int = 200,
        populate_buffer: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Collect data from one episode.
        
        Args:
            env: Gymnasium environment
            max_steps: Maximum steps to collect
            populate_buffer: If True and experience_buffer exists, add data to buffer
            
        Returns:
            Dictionary containing observations, actions, rewards, and next_observations
        """
        self.model.eval()
        
        obs, _ = env.reset()
        
        # Initialize hidden state if model is stateful
        if hasattr(self.model, 'init_hidden'):
            hidden_state = self.model.init_hidden(batch_size=1)
        else:
            hidden_state = None
        
        observations = []
        actions = []
        rewards = []
        next_observations = []
        
        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if hidden_state is not None:
                    model_output, hidden_state = self.model(obs_tensor, hidden_state)
                else:
                    model_output = self.model(obs_tensor)
                
                # For state prediction models, project to action space
                if model_output.shape[-1] == self.state_dim:
                    action_logits = model_output[:, :self.action_dim]
                else:
                    action_logits = model_output
                
                # Sample action (for continuous control, use tanh squashing)
                action = torch.tanh(action_logits).cpu().numpy().flatten()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            
            # Add to experience buffer if requested
            if populate_buffer and self.experience_buffer is not None:
                obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
                self.experience_buffer.add(obs_t.unsqueeze(0), next_obs_t.unsqueeze(0))
            
            obs = next_obs
            
            if done:
                break
        
        # Convert to tensors (1, T, D) format
        return {
            'observations': torch.tensor(np.array(observations), dtype=torch.float32).unsqueeze(0).to(self.device),
            'actions': torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(0).to(self.device),
            'rewards': torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device),
            'next_observations': torch.tensor(np.array(next_observations), dtype=torch.float32).unsqueeze(0).to(self.device)
        }
    
    def meta_train_step(self, task_id: int) -> float:
        """Perform one meta-training step on a task."""
        env = self.task_dist.sample_task(task_id)
        
        # Collect support and query data (populate buffer during training)
        support_data = self.collect_episode_data(
            env, 
            max_steps=self.config.get('support_steps', 100),
            populate_buffer=True
        )
        query_data = self.collect_episode_data(
            env, 
            max_steps=self.config.get('query_steps', 100),
            populate_buffer=True
        )
        
        env.close()
        
        # Prepare data for MetaMAML
        obs_support = support_data['observations']
        next_obs_support = support_data['next_observations']
        obs_query = query_data['observations']
        next_obs_query = query_data['next_observations']
        
        # Check if we have enough data
        if obs_support.shape[1] < 2 or obs_query.shape[1] < 2:
            return 0.0
        
        # Create tasks list
        tasks = [(obs_support, next_obs_support, obs_query, next_obs_query)]
        
        # Initialize hidden state if needed
        if hasattr(self.model, 'init_hidden'):
            initial_hidden = self.model.init_hidden(batch_size=1)
        else:
            initial_hidden = None
        
        # Meta-update
        meta_loss = self.meta_learner.meta_update(
            tasks=tasks,
            initial_hidden_state=initial_hidden,
            loss_fn=nn.MSELoss()
        )
        
        return meta_loss
    
    def meta_test(self, task_id: int, num_adapt_steps: int = 5) -> Dict[str, float]:
        """Test adaptation on a held-out task."""
        env = self.task_dist.sample_task(task_id)
        
        # Collect adaptation data (don't populate buffer during testing)
        adapt_data = self.collect_episode_data(
            env, 
            max_steps=50,
            populate_buffer=False
        )
        
        # Create adapter based on adaptation mode
        adaptation_mode = self.config.get('adaptation_mode', 'standard')
        
        if adaptation_mode == 'hybrid':
            if self.experience_buffer is None:
                raise ValueError("ExperienceBuffer is required for hybrid adaptation mode")
            
            adapt_config = HybridAdaptationConfig(
                learning_rate=self.config.get('adapt_lr', 0.01),
                num_steps=num_adapt_steps,
                experience_batch_size=self.config.get('experience_batch_size', 32),
                experience_weight=self.config.get('experience_weight', 0.1)
            )
            adapter = HybridAdapter(
                model=self.model,
                config=adapt_config,
                experience_buffer=self.experience_buffer,
                device=self.device
            )
        else:  # standard mode
            adapt_config = StandardAdaptationConfig(
                learning_rate=self.config.get('adapt_lr', 0.01),
                num_steps=num_adapt_steps
            )
            adapter = StandardAdapter(
                model=self.model,
                config=adapt_config,
                device=self.device
            )
        
        # Perform adaptation
        obs = adapt_data['observations']
        next_obs = adapt_data['next_observations']
        
        adaptation_losses = []
        
        if hasattr(self.model, 'init_hidden'):
            hidden_state = self.model.init_hidden(batch_size=1)
        else:
            hidden_state = None
        
        # Adapt on each timestep
        for t in range(min(obs.shape[1], num_adapt_steps)):
            x = obs[:, t, :]
            y = next_obs[:, t, :]
            
            if hidden_state is not None:
                loss, _ = adapter.update_step(
                    x_current=x,
                    y_current=y,
                    hidden_state_current=hidden_state
                )
                # Update hidden state
                with torch.no_grad():
                    _, hidden_state = self.model(x, hidden_state)
                    hidden_state = hidden_state.detach()
            else:
                # For stateless models
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(x)
                    loss = nn.MSELoss()(pred, y).item()
            
            adaptation_losses.append(loss)
        
        # Evaluate final performance
        eval_data = self.collect_episode_data(env, max_steps=200, populate_buffer=False)
        final_reward = eval_data['rewards'].sum().item()
        
        env.close()
        
        return {
            'initial_loss': adaptation_losses[0] if adaptation_losses else 0.0,
            'final_loss': adaptation_losses[-1] if adaptation_losses else 0.0,
            'adaptation_losses': adaptation_losses,
            'final_reward': final_reward,
            'num_steps': len(adaptation_losses)
        }
    
    def run(self, num_epochs: int = 50, eval_interval: int = 10):
        """Run the full benchmark."""
        print(f"\n{'='*70}")
        print(f"Running Benchmark: {self.task_dist_name} with {self.method_name.upper()}")
        print(f"Adaptation Mode: {self.config.get('adaptation_mode', 'standard').upper()}")
        print(f"{'='*70}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"Num tasks: {self.task_dist.num_tasks}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.experience_buffer is not None:
            print(f"Experience buffer size: {self.experience_buffer.max_size}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Meta-training loop
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Sample tasks for this epoch
            task_ids = np.random.choice(
                self.task_dist.num_tasks,
                size=self.config.get('tasks_per_epoch', 5),
                replace=True
            )
            
            for task_id in task_ids:
                loss = self.meta_train_step(int(task_id))
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            self.metrics['train_loss'].append(avg_loss)
            self.metrics['epoch'].append(epoch)
            
            # Evaluation
            if (epoch + 1) % eval_interval == 0:
                buffer_info = ""
                if self.experience_buffer is not None:
                    buffer_info = f" | Buffer: {len(self.experience_buffer)}"
                
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}{buffer_info}")
                
                # Test on held-out task
                test_task_id = self.task_dist.num_tasks - 1
                test_results = self.meta_test(test_task_id, num_adapt_steps=10)
                
                print(f"  Test - Initial Loss: {test_results['initial_loss']:.4f}, "
                      f"Final Loss: {test_results['final_loss']:.4f}, "
                      f"Reward: {test_results['final_reward']:.2f}")
                
                self.metrics['test_initial_loss'].append(test_results['initial_loss'])
                self.metrics['test_final_loss'].append(test_results['final_loss'])
                self.metrics['test_reward'].append(test_results['final_reward'])
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"Benchmark completed in {elapsed_time:.2f} seconds")
        print(f"{'='*70}\n")
        
        return self.metrics


def main():
    parser = argparse.ArgumentParser(
        description="SSM-MetaRL-Unified Serious Benchmark Suite"
    )
    
    # Task and method selection
    parser.add_argument('--task', type=str, default='cartpole-simple',
                       help='Task distribution name')
    parser.add_argument('--method', type=str, default='ssm',
                       choices=['ssm', 'mlp', 'lstm', 'gru', 'transformer', 'all'],
                       help='Method to benchmark')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of meta-training epochs')
    parser.add_argument('--tasks_per_epoch', type=int, default=5,
                       help='Number of tasks to sample per epoch')
    parser.add_argument('--eval_interval', type=int, default=10,
                       help='Evaluation interval in epochs')
    
    # Model settings
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for models')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                       help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                       help='Outer loop learning rate')
    parser.add_argument('--adapt_lr', type=float, default=0.01,
                       help='Adaptation learning rate')
    
    # Adaptation settings
    parser.add_argument('--adaptation_mode', type=str, default='standard',
                       choices=['standard', 'hybrid'],
                       help='Adaptation mode: standard or hybrid (with experience replay)')
    
    # Experience replay settings (for hybrid mode)
    parser.add_argument('--buffer_size', type=int, default=10000,
                       help='Maximum size of experience buffer')
    parser.add_argument('--experience_batch_size', type=int, default=32,
                       help='Batch size for sampling from experience buffer')
    parser.add_argument('--experience_weight', type=float, default=0.1,
                       help='Weight for experience loss in hybrid mode')
    
    # Episode settings
    parser.add_argument('--support_steps', type=int, default=100,
                       help='Steps for support set collection')
    parser.add_argument('--query_steps', type=int, default=100,
                       help='Steps for query set collection')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for training')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare config
    config = {
        'device': args.device,
        'hidden_dim': args.hidden_dim,
        'inner_lr': args.inner_lr,
        'outer_lr': args.outer_lr,
        'adapt_lr': args.adapt_lr,
        'tasks_per_epoch': args.tasks_per_epoch,
        'support_steps': args.support_steps,
        'query_steps': args.query_steps,
        'adaptation_mode': args.adaptation_mode,
        'buffer_size': args.buffer_size,
        'experience_batch_size': args.experience_batch_size,
        'experience_weight': args.experience_weight,
    }
    
    # Run benchmarks
    methods = ['ssm', 'mlp', 'lstm', 'gru', 'transformer'] if args.method == 'all' else [args.method]
    
    all_results = {}
    
    for method in methods:
        print(f"\n{'#'*70}")
        print(f"# Benchmarking method: {method.upper()}")
        print(f"{'#'*70}\n")
        
        try:
            runner = BenchmarkRunner(args.task, method, config)
            metrics = runner.run(num_epochs=args.epochs, eval_interval=args.eval_interval)
            all_results[method] = metrics
            
            # Save results
            result_file = output_dir / f"{args.task}_{method}_{args.adaptation_mode}_results.json"
            with open(result_file, 'w') as f:
                json.dump({k: [float(v) if isinstance(v, (np.floating, float)) else v for v in vals] 
                          for k, vals in metrics.items()}, f, indent=2)
            
            print(f"Results saved to {result_file}")
            
        except Exception as e:
            print(f"Error running benchmark for {method}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("All benchmarks completed!")
    print(f"Results saved to {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

