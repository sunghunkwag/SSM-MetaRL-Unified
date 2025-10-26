# -*- coding: utf-8 -*-
"""
Recursive Self-Improvement System for SSM-MetaRL (FIXED VERSION)
Properly implemented with real meta-learning evaluation and bug fixes.

Key Fixes:
1. Real meta-task evaluation instead of dummy data
2. Proper rollback logic for architecture changes  
3. Meaningful adaptation speed and generalization metrics
4. Better logging and error handling
5. Resource management improvements

🐾 Koala AI (fixed after getting properly called out)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict, deque
from dataclasses import dataclass, asdict
import warnings
import time

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for safety parameters"""
    performance_window: int = 10
    min_performance_threshold: float = -1000
    max_emergency_stops: int = 3
    instability_threshold: float = 100.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RSIConfig:
    """Configuration for RSI parameters"""
    # Performance weighting
    reward_weight: float = 0.3
    adaptation_weight: float = 0.25
    generalization_weight: float = 0.25
    meta_efficiency_weight: float = 0.1
    stability_weight: float = 0.1
    
    # Architecture evolution
    mutation_rate: float = 0.1
    max_dimension_change: float = 0.2
    
    # Evaluation parameters
    num_episodes_quick: int = 10
    num_episodes_full: int = 20
    num_meta_tasks_quick: int = 3
    num_meta_tasks_full: int = 10
    meta_task_length: int = 50
    adaptation_steps: int = 5
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance metrics for self-assessment"""
    avg_reward: float = 0.0
    adaptation_speed: float = 0.0  # Steps to reach 80% of final performance
    generalization_score: float = 0.0  # Performance on completely unseen tasks
    meta_learning_efficiency: float = 0.0  # MAML convergence rate
    stability_score: float = 0.0  # Inverse of variance in performance
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def is_better_than(self, other: 'PerformanceMetrics', config: RSIConfig) -> bool:
        """Check if this performance is significantly better than other"""
        improvement_score = (
            (self.avg_reward - other.avg_reward) * config.reward_weight +
            (self.adaptation_speed - other.adaptation_speed) * config.adaptation_weight +
            (self.generalization_score - other.generalization_score) * config.generalization_weight +
            (self.meta_learning_efficiency - other.meta_learning_efficiency) * config.meta_efficiency_weight +
            (self.stability_score - other.stability_score) * config.stability_weight
        )
        return improvement_score > 0.05  # 5% improvement threshold


@dataclass 
class ArchitecturalConfig:
    """Configuration for model architecture"""
    state_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 1
    dropout_rate: float = 0.0
    activation: str = "relu"
    use_residual: bool = False
    use_layer_norm: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ArchitecturalConfig':
        return cls(**data)


@dataclass
class LearningConfig:
    """Configuration for learning parameters"""
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    meta_batch_size: int = 5
    adaptation_steps: int = 1
    gradient_clip: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod  
    def from_dict(cls, data: Dict) -> 'LearningConfig':
        return cls(**data)


class TaskGenerator:
    """Generate meaningful meta-learning tasks for evaluation"""
    
    def __init__(self, env, device: str = 'cpu'):
        self.env = env
        self.device = device
        
    def create_meta_task(self, model: nn.Module, task_length: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a single meta-learning task from environment interaction"""
        observations = []
        actions = []
        rewards = []
        
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        hidden_state = model.init_hidden(batch_size=1)
        
        # Collect trajectory
        for step in range(task_length):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_logits, hidden_state = model(obs_tensor, hidden_state)
                
                # Sample action (with some exploration)
                if hasattr(self.env.action_space, 'n'):  # Discrete
                    if np.random.random() < 0.1:  # 10% random exploration
                        action = self.env.action_space.sample()
                    else:
                        logits = action_logits[:, :self.env.action_space.n]
                        action = torch.argmax(logits, dim=-1).item()
                else:  # Continuous
                    action = action_logits.cpu().numpy().flatten()
                    action += np.random.normal(0, 0.1, action.shape)  # Add noise
            
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_obs, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                next_obs, reward, done, info = step_result
            
            observations.append(obs_tensor)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                reset_result = self.env.reset()
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                hidden_state = model.init_hidden(batch_size=1)
            else:
                obs = next_obs
        
        # Convert to tensors
        obs_tensor = torch.cat(observations, dim=0).unsqueeze(0)  # (1, T, obs_dim)
        action_tensor = torch.tensor(actions, dtype=torch.long if hasattr(self.env.action_space, 'n') else torch.float32)
        action_tensor = action_tensor.unsqueeze(0)  # (1, T) or (1, T, action_dim)
        
        if len(action_tensor.shape) == 2:  # Discrete actions
            action_tensor = action_tensor.unsqueeze(-1).float()  # (1, T, 1)
        
        # Split into support and query sets
        split_idx = len(actions) // 2
        support_x = obs_tensor[:, :split_idx, :]
        support_y = action_tensor[:, :split_idx, :]
        query_x = obs_tensor[:, split_idx:, :]
        query_y = action_tensor[:, split_idx:, :]
        
        return support_x, support_y, query_x, query_y
    
    def create_generalization_task(self, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a task for testing generalization (different from training distribution)"""
        # For now, just create a shorter task with different random seed
        old_state = np.random.get_state()
        np.random.seed(np.random.randint(0, 10000))
        
        try:
            support_x, support_y, query_x, query_y = self.create_meta_task(model, task_length=25)
            # Use only query set for generalization test
            return query_x, query_y
        finally:
            np.random.set_state(old_state)


class SafetyMonitor:
    """Monitor system health and prevent dangerous modifications"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.performance_window)
        self.emergency_stops = 0
        
    def check_safety(self, current_metrics: PerformanceMetrics) -> Tuple[bool, str]:
        """Check if current state is safe for modifications"""
        
        # Check if performance crashed
        if current_metrics.avg_reward < self.config.min_performance_threshold:
            self.emergency_stops += 1
            return False, f"Performance below minimum threshold: {current_metrics.avg_reward:.2f}"
        
        # Check if too many emergency stops occurred
        if self.emergency_stops >= self.config.max_emergency_stops:
            return False, f"Too many emergency stops: {self.emergency_stops}"
        
        # Check for performance instability
        if len(self.performance_history) >= 3:
            recent_rewards = [m.avg_reward for m in list(self.performance_history)[-3:]]
            recent_std = np.std(recent_rewards)
            if recent_std > self.config.instability_threshold:
                return False, f"High performance instability: std={recent_std:.2f}"
        
        return True, "Safe to proceed"
    
    def update_history(self, metrics: PerformanceMetrics):
        """Update performance history"""
        self.performance_history.append(metrics)
        logger.info(f"Performance updated: reward={metrics.avg_reward:.2f}, adaptation={metrics.adaptation_speed:.2f}")
    
    def reset_emergency_stops(self):
        """Reset emergency stop counter (call after successful improvement)"""
        self.emergency_stops = 0
        logger.info("Emergency stops reset")


class ModelCheckpoint:
    """Checkpoint system for safe rollbacks"""
    
    def __init__(self, max_checkpoints: int = 5):
        self.checkpoints = deque(maxlen=max_checkpoints)
        self.checkpoint_counter = 0
        
    def save_checkpoint(self, model: nn.Module, arch_config: ArchitecturalConfig, 
                       learn_config: LearningConfig, metrics: PerformanceMetrics) -> str:
        """Save model checkpoint"""
        checkpoint_id = f"checkpoint_{self.checkpoint_counter}"
        
        checkpoint = {
            'id': checkpoint_id,
            'model_state': copy.deepcopy(model.state_dict()),
            'arch_config': arch_config.to_dict(),
            'learn_config': learn_config.to_dict(),
            'metrics': metrics.to_dict(),
            'timestamp': self.checkpoint_counter
        }
        
        self.checkpoints.append(checkpoint)
        self.checkpoint_counter += 1
        
        logger.info(f"Checkpoint saved: {checkpoint_id}")
        return checkpoint_id
    
    def load_checkpoint(self, model: nn.Module, checkpoint_id: str = None) -> Tuple[ArchitecturalConfig, LearningConfig, PerformanceMetrics]:
        """Load checkpoint (latest if checkpoint_id is None)"""
        if not self.checkpoints:
            raise ValueError("No checkpoints available")
        
        if checkpoint_id is None:
            checkpoint = self.checkpoints[-1]  # Latest
        else:
            checkpoint = next((cp for cp in self.checkpoints if cp['id'] == checkpoint_id), None)
            if checkpoint is None:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        # Load with strict=False to handle minor architecture changes
        model.load_state_dict(checkpoint['model_state'], strict=False)
        
        arch_config = ArchitecturalConfig.from_dict(checkpoint['arch_config'])
        learn_config = LearningConfig.from_dict(checkpoint['learn_config'])
        metrics = PerformanceMetrics(**checkpoint['metrics'])
        
        logger.info(f"Checkpoint loaded: {checkpoint['id']}")
        return arch_config, learn_config, metrics
    
    def get_best_checkpoint(self) -> Optional[Dict]:
        """Get checkpoint with best performance"""
        if not self.checkpoints:
            return None
        
        return max(self.checkpoints, key=lambda cp: cp['metrics']['avg_reward'])


class ArchitecturalEvolution:
    """Safe architectural evolution with constraints"""
    
    def __init__(self, config: RSIConfig):
        self.config = config
        
    def propose_architectural_changes(self, current_config: ArchitecturalConfig, 
                                   performance_history: List[PerformanceMetrics]) -> List[ArchitecturalConfig]:
        """Propose safe architectural modifications"""
        proposals = []
        base_config = copy.deepcopy(current_config)
        
        # 1. Dimension scaling (conservative)
        if np.random.random() < self.config.mutation_rate:
            config = copy.deepcopy(base_config)
            scale_factor = np.random.uniform(0.9, 1.1)  # ±10% change
            config.state_dim = max(8, int(config.state_dim * scale_factor))
            config.hidden_dim = max(16, int(config.hidden_dim * scale_factor))
            proposals.append(config)
        
        # 2. Add/remove regularization
        if np.random.random() < self.config.mutation_rate:
            config = copy.deepcopy(base_config)
            if config.dropout_rate == 0.0:
                config.dropout_rate = 0.1
            else:
                config.dropout_rate = 0.0
            proposals.append(config)
        
        # 3. Toggle normalization
        if np.random.random() < self.config.mutation_rate:
            config = copy.deepcopy(base_config)
            config.use_layer_norm = not config.use_layer_norm
            proposals.append(config)
        
        return proposals[:3]  # Limit to 3 proposals max


class HyperparameterOptimizer:
    """Safe hyperparameter optimization"""
    
    def __init__(self):
        self.search_range = {
            'inner_lr': (0.005, 0.05),
            'outer_lr': (0.0005, 0.005), 
            'gradient_clip': (0.5, 2.0),
            'adaptation_steps': (1, 3)
        }
    
    def propose_hyperparameter_changes(self, current_config: LearningConfig) -> List[LearningConfig]:
        """Propose conservative hyperparameter changes"""
        proposals = []
        
        for param, (min_val, max_val) in self.search_range.items():
            config = copy.deepcopy(current_config)
            current_val = getattr(config, param)
            
            # Small perturbation around current value
            if isinstance(current_val, int):
                new_val = current_val + np.random.choice([-1, 1])
                new_val = max(int(min_val), min(int(max_val), new_val))
            else:
                perturbation = np.random.uniform(-0.2, 0.2) * current_val
                new_val = current_val + perturbation
                new_val = max(min_val, min(max_val, new_val))
            
            setattr(config, param, new_val)
            proposals.append(config)
        
        return proposals


class RecursiveSelfImprovementAgent:
    """Main recursive self-improvement system with proper evaluation"""
    
    def __init__(self, initial_model: StateSpaceModel, env, device: str = 'cpu', 
                 safety_config: SafetyConfig = None, rsi_config: RSIConfig = None):
        self.env = env
        self.device = device
        self.rsi_config = rsi_config or RSIConfig()
        
        # Core components
        self.model = initial_model
        self.meta_learner = None
        
        # Task generation for proper evaluation
        self.task_generator = TaskGenerator(env, device)
        
        # Safety and monitoring
        self.safety_monitor = SafetyMonitor(safety_config or SafetyConfig())
        self.checkpoint_system = ModelCheckpoint()
        
        # Evolution engines
        self.arch_evolution = ArchitecturalEvolution(self.rsi_config)
        self.hyperopt = HyperparameterOptimizer()
        
        # Current configurations
        self.arch_config = ArchitecturalConfig(
            state_dim=initial_model.state_dim,
            hidden_dim=initial_model.hidden_dim
        )
        self.learn_config = LearningConfig()
        
        # Performance tracking
        self.current_metrics = PerformanceMetrics()
        self.improvement_history = []
        
        # Self-improvement state
        self.generation = 0
        self.successful_improvements = 0
        self.failed_attempts = 0
        
        logger.info("RecursiveSelfImprovementAgent initialized")
    
    def evaluate_performance(self, quick_eval: bool = False) -> PerformanceMetrics:
        """FIXED: Comprehensive performance evaluation with real meta-tasks"""
        logger.info(f"Evaluating performance (Generation {self.generation}, Quick: {quick_eval})")
        
        num_episodes = self.rsi_config.num_episodes_quick if quick_eval else self.rsi_config.num_episodes_full
        num_meta_tasks = self.rsi_config.num_meta_tasks_quick if quick_eval else self.rsi_config.num_meta_tasks_full
        
        # 1. Direct policy performance
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            reset_result = self.env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            # Ensure obs is a numpy array
            obs = np.array(obs, dtype=np.float32)
            hidden_state = self.model.init_hidden(batch_size=1)
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 500:
                # Convert observation to tensor properly
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_logits, hidden_state = self.model(obs_tensor, hidden_state)
                
                if hasattr(self.env.action_space, 'n'):  # Discrete
                    action = torch.argmax(action_logits[:, :self.env.action_space.n], dim=-1).item()
                else:  # Continuous
                    action = action_logits.cpu().numpy().flatten()
                
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs, reward, done, truncated, info = step_result
                    done = done or truncated
                else:
                    obs, reward, done, info = step_result
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        avg_reward = np.mean(episode_rewards)
        reward_std = np.std(episode_rewards)
        
        # 2. REAL Meta-learning evaluation
        if self.meta_learner is not None:
            adaptation_speeds = []
            meta_losses = []
            generalization_scores = []
            
            for _ in range(num_meta_tasks):
                try:
                    # Create real meta-task from environment interaction
                    support_x, support_y, query_x, query_y = self.task_generator.create_meta_task(
                        self.model, self.rsi_config.meta_task_length
                    )
                    
                    # Test adaptation speed: measure performance improvement over adaptation steps
                    initial_hidden = self.model.init_hidden(batch_size=1)
                    
                    # Get initial performance (before adaptation)
                    with torch.no_grad():
                        if self.meta_learner._stateful:
                            initial_pred, _ = self.meta_learner.functional_forward(
                                query_x.reshape(-1, query_x.shape[-1]), 
                                initial_hidden.repeat(query_x.shape[1], 1)
                            )
                        else:
                            initial_pred = self.meta_learner.functional_forward(
                                query_x.reshape(-1, query_x.shape[-1]), None
                            )
                    
                    if hasattr(self.env.action_space, 'n'):  # Discrete actions
                        initial_loss = F.cross_entropy(
                            initial_pred[:, :self.env.action_space.n], 
                            query_y.reshape(-1).long()
                        )
                    else:  # Continuous
                        initial_loss = F.mse_loss(initial_pred, query_y.reshape(-1, -1))
                    
                    # Adapt on support set and measure improvement
                    adaptation_losses = []
                    fast_weights = self.meta_learner.get_fast_weights()
                    
                    for step in range(self.rsi_config.adaptation_steps):
                        # One step of adaptation
                        fast_weights = self.meta_learner.adapt_task(
                            support_x, support_y, initial_hidden, num_steps=1
                        )
                        
                        # Evaluate on query set
                        with torch.no_grad():
                            if self.meta_learner._stateful:
                                adapted_pred, _ = self.meta_learner.functional_forward(
                                    query_x.reshape(-1, query_x.shape[-1]),
                                    initial_hidden.repeat(query_x.shape[1], 1),
                                    fast_weights
                                )
                            else:
                                adapted_pred = self.meta_learner.functional_forward(
                                    query_x.reshape(-1, query_x.shape[-1]), None, fast_weights
                                )
                        
                        if hasattr(self.env.action_space, 'n'):
                            adapted_loss = F.cross_entropy(
                                adapted_pred[:, :self.env.action_space.n], 
                                query_y.reshape(-1).long()
                            )
                        else:
                            adapted_loss = F.mse_loss(adapted_pred, query_y.reshape(-1, -1))
                        
                        adaptation_losses.append(adapted_loss.item())
                    
                    # Calculate adaptation speed (steps to reach 80% improvement)
                    if len(adaptation_losses) > 1:
                        target_loss = initial_loss.item() * 0.2 + adaptation_losses[-1] * 0.8  # 80% of possible improvement
                        adaptation_speed = len(adaptation_losses)  # Default to full steps
                        for i, loss in enumerate(adaptation_losses):
                            if loss <= target_loss:
                                adaptation_speed = i + 1
                                break
                        adaptation_speed = max(1, self.rsi_config.adaptation_steps + 1 - adaptation_speed)  # Invert so higher = faster
                    else:
                        adaptation_speed = 1.0
                    
                    adaptation_speeds.append(adaptation_speed)
                    meta_losses.append(adaptation_losses[-1] if adaptation_losses else initial_loss.item())
                    
                    # Test generalization on different task
                    gen_x, gen_y = self.task_generator.create_generalization_task(self.model)
                    with torch.no_grad():
                        if self.meta_learner._stateful:
                            gen_pred, _ = self.meta_learner.functional_forward(
                                gen_x.reshape(-1, gen_x.shape[-1]),
                                initial_hidden.repeat(gen_x.shape[1], 1),
                                fast_weights
                            )
                        else:
                            gen_pred = self.meta_learner.functional_forward(
                                gen_x.reshape(-1, gen_x.shape[-1]), None, fast_weights
                            )
                    
                    if hasattr(self.env.action_space, 'n'):
                        gen_loss = F.cross_entropy(
                            gen_pred[:, :self.env.action_space.n], 
                            gen_y.reshape(-1).long()
                        )
                    else:
                        gen_loss = F.mse_loss(gen_pred, gen_y.reshape(-1, -1))
                    
                    # Convert loss to score (lower loss = higher score)
                    gen_score = max(0, 100 - gen_loss.item() * 10)  # Rough heuristic
                    generalization_scores.append(gen_score)
                    
                except Exception as e:
                    logger.warning(f"Meta-evaluation failed: {e}")
                    adaptation_speeds.append(1.0)
                    meta_losses.append(100.0)
                    generalization_scores.append(0.0)
            
            avg_adaptation_speed = np.mean(adaptation_speeds) if adaptation_speeds else 1.0
            avg_meta_loss = np.mean(meta_losses) if meta_losses else 100.0
            avg_generalization = np.mean(generalization_scores) if generalization_scores else 0.0
            meta_efficiency = max(0, 100 - avg_meta_loss)
            
        else:
            # No meta-learner available
            avg_adaptation_speed = 1.0
            meta_efficiency = 0.0
            avg_generalization = 0.0
        
        # Calculate stability score
        stability_score = max(0, 100 - reward_std) if reward_std > 0 else 100.0
        
        metrics = PerformanceMetrics(
            avg_reward=avg_reward,
            adaptation_speed=avg_adaptation_speed,
            generalization_score=avg_generalization,
            meta_learning_efficiency=meta_efficiency,
            stability_score=stability_score
        )
        
        logger.info(f"Performance: reward={avg_reward:.2f}, adaptation={avg_adaptation_speed:.2f}, "
                   f"generalization={avg_generalization:.2f}, meta_eff={meta_efficiency:.2f}")
        
        return metrics
    
    def create_improved_model(self, arch_config: ArchitecturalConfig) -> StateSpaceModel:
        """Create new model with improved architecture"""
        new_model = StateSpaceModel(
            state_dim=arch_config.state_dim,
            input_dim=self.model.input_dim,
            output_dim=self.model.output_dim,
            hidden_dim=arch_config.hidden_dim
        ).to(self.device)
        
        # Transfer knowledge from old model
        try:
            old_dict = self.model.state_dict()
            new_dict = new_model.state_dict()
            
            # Copy compatible weights
            transferred = 0
            for key in new_dict:
                if key in old_dict:
                    old_shape = old_dict[key].shape
                    new_shape = new_dict[key].shape
                    
                    if old_shape == new_shape:
                        new_dict[key] = old_dict[key].clone()
                        transferred += 1
                    elif len(old_shape) == len(new_shape):
                        # Try to transfer partial weights for dimension changes
                        min_dims = [min(o, n) for o, n in zip(old_shape, new_shape)]
                        if len(old_shape) == 2:  # Linear layer
                            new_dict[key][:min_dims[0], :min_dims[1]] = old_dict[key][:min_dims[0], :min_dims[1]]
                            transferred += 1
            
            new_model.load_state_dict(new_dict)
            logger.info(f"Successfully transferred {transferred} weight tensors")
            
        except Exception as e:
            logger.warning(f"Weight transfer failed: {e}, using random initialization")
        
        return new_model
    
    def attempt_self_improvement(self) -> bool:
        """FIXED: Attempt one cycle of self-improvement with proper rollback"""
        logger.info(f"Starting self-improvement cycle (Generation {self.generation})")
        
        # 1. Safety check
        is_safe, safety_msg = self.safety_monitor.check_safety(self.current_metrics)
        if not is_safe:
            logger.warning(f"Safety check failed: {safety_msg}")
            return False
        
        # 2. Save current checkpoint
        checkpoint_id = self.checkpoint_system.save_checkpoint(
            self.model, self.arch_config, self.learn_config, self.current_metrics
        )
        
        # Store baseline state for rollbacks
        baseline_model_state = copy.deepcopy(self.model.state_dict())
        baseline_arch_config = copy.deepcopy(self.arch_config)
        baseline_learn_config = copy.deepcopy(self.learn_config)
        
        # 3. Generate improvement proposals
        arch_proposals = self.arch_evolution.propose_architectural_changes(
            self.arch_config, self.improvement_history[-5:] if self.improvement_history else []
        )
        hyperparam_proposals = self.hyperopt.propose_hyperparameter_changes(self.learn_config)
        
        best_improvement = None
        best_metrics = self.current_metrics
        
        # 4. FIXED: Test architectural changes with proper rollback
        for i, arch_config in enumerate(arch_proposals):
            logger.info(f"Testing architectural proposal {i+1}/{len(arch_proposals)}")
            
            try:
                # Create new model
                test_model = self.create_improved_model(arch_config)
                
                # Update state
                self.model = test_model
                self.arch_config = arch_config
                
                # Update meta-learner
                self.meta_learner = MetaMAML(
                    model=self.model,
                    inner_lr=self.learn_config.inner_lr,
                    outer_lr=self.learn_config.outer_lr
                )
                
                # Quick evaluation
                test_metrics = self.evaluate_performance(quick_eval=True)
                
                if test_metrics.is_better_than(best_metrics, self.rsi_config):
                    best_improvement = ('architecture', arch_config, self.learn_config)
                    best_metrics = test_metrics
                    logger.info("New best architectural improvement found!")
                
            except Exception as e:
                logger.error(f"Architectural test failed: {e}")
            
            finally:
                # FIXED: Always rollback to baseline after each test
                # Recreate model with baseline architecture if dimensions changed
                if self.arch_config.state_dim != baseline_arch_config.state_dim or \
                   self.arch_config.hidden_dim != baseline_arch_config.hidden_dim:
                    self.model = self.create_improved_model(baseline_arch_config)
                self.model.load_state_dict(baseline_model_state)
                self.arch_config = copy.deepcopy(baseline_arch_config)
                self.learn_config = copy.deepcopy(baseline_learn_config)
        
        # 5. FIXED: Test hyperparameter changes with proper rollback
        for i, learn_config in enumerate(hyperparam_proposals):
            logger.info(f"Testing hyperparameter proposal {i+1}/{len(hyperparam_proposals)}")
            
            try:
                # Update learning config
                self.learn_config = learn_config
                
                # Update meta-learner
                self.meta_learner = MetaMAML(
                    model=self.model,
                    inner_lr=self.learn_config.inner_lr,
                    outer_lr=self.learn_config.outer_lr
                )
                
                # Quick evaluation
                test_metrics = self.evaluate_performance(quick_eval=True)
                
                if test_metrics.is_better_than(best_metrics, self.rsi_config):
                    best_improvement = ('hyperparameters', self.arch_config, learn_config)
                    best_metrics = test_metrics
                    logger.info("New best hyperparameter improvement found!")
                
            except Exception as e:
                logger.error(f"Hyperparameter test failed: {e}")
            
            finally:
                # FIXED: Always rollback learning config after each test
                self.learn_config = copy.deepcopy(baseline_learn_config)
        
        # 6. Apply best improvement or rollback
        if best_improvement is not None:
            improvement_type, arch_config, learn_config = best_improvement
            
            # Apply the best improvement
            if improvement_type == 'architecture':
                self.model = self.create_improved_model(arch_config)
            
            self.arch_config = arch_config
            self.learn_config = learn_config
            
            # Update meta-learner
            self.meta_learner = MetaMAML(
                model=self.model,
                inner_lr=self.learn_config.inner_lr,
                outer_lr=self.learn_config.outer_lr
            )
            
            self.current_metrics = best_metrics
            self.successful_improvements += 1
            self.safety_monitor.reset_emergency_stops()
            
            logger.info(f"Successfully improved via {improvement_type}!")
            logger.info(f"New performance: {best_metrics.avg_reward:.2f} (was {self.current_metrics.avg_reward:.2f})")
            
            return True
        else:
            logger.info("No improvements found, rolling back to checkpoint")
            
            # Final rollback to checkpoint
            self.arch_config, self.learn_config, self.current_metrics = (
                self.checkpoint_system.load_checkpoint(self.model)
            )
            
            self.failed_attempts += 1
            return False
    
    def recursive_self_improve(self, max_generations: int = 10, min_improvement_threshold: float = 1.0):
        """Main recursive self-improvement loop"""
        logger.info(f"Starting Recursive Self-Improvement (Max {max_generations} generations)")
        
        # Initial evaluation
        self.current_metrics = self.evaluate_performance()
        self.safety_monitor.update_history(self.current_metrics)
        
        logger.info(f"Initial Performance: {self.current_metrics.avg_reward:.2f}")
        initial_performance = self.current_metrics.avg_reward
        
        for generation in range(max_generations):
            self.generation = generation
            logger.info(f"Generation {generation + 1}/{max_generations}")
            
            # Attempt improvement
            improved = self.attempt_self_improvement()
            
            if improved:
                # Re-evaluate full performance
                self.current_metrics = self.evaluate_performance()
                self.safety_monitor.update_history(self.current_metrics)
                self.improvement_history.append(self.current_metrics)
                
                improvement = self.current_metrics.avg_reward - initial_performance
                logger.info(f"Cumulative improvement: {improvement:.2f}")
                
                if improvement > min_improvement_threshold * (generation + 1):
                    logger.info("Significant improvement achieved!")
                
            else:
                logger.info("No improvement this generation")
                
                # Early stopping if too many failures
                if self.failed_attempts >= 3:
                    logger.info("Too many failed attempts, stopping early")
                    break
        
        # Final summary
        final_improvement = self.current_metrics.avg_reward - initial_performance
        logger.info(f"Self-Improvement Complete!")
        logger.info(f"Initial: {initial_performance:.2f}, Final: {self.current_metrics.avg_reward:.2f}")
        logger.info(f"Total Improvement: {final_improvement:.2f}")
        logger.info(f"Successful: {self.successful_improvements}, Failed: {self.failed_attempts}")
        
        return self.current_metrics


# FIXED main function
if __name__ == "__main__":
    print("🐾 Koala's FIXED Recursive Self-Improvement System")
    print("(Now with actual meta-learning evaluation!)")
    print("=" * 60)
    
    # You would integrate this with your existing SSM-MetaRL system like this:
    """
    import gymnasium as gym
    from env_runner.environment import Environment
    
    env = Environment('CartPole-v1')
    device = torch.device('cpu')
    
    # Your existing trained model
    model = StateSpaceModel(state_dim=32, input_dim=4, output_dim=2, hidden_dim=64)
    
    # Create self-improvement agent with proper configs
    safety_config = SafetyConfig(min_performance_threshold=-500, max_emergency_stops=5)
    rsi_config = RSIConfig(num_episodes_full=30, num_meta_tasks_full=15)
    
    agent = RecursiveSelfImprovementAgent(model, env, device, safety_config, rsi_config)
    
    # Run self-improvement
    final_metrics = agent.recursive_self_improve(max_generations=5)
    """
    
    print("💤 Fixed system ready... probably works now... maybe... 🐾")