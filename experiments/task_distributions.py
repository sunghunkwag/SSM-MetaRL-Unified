"""
High-Dimensional Task Distributions for Meta-RL Benchmarking

This module provides task distribution wrappers for creating meta-learning
benchmarks with MuJoCo environments. Each distribution creates multiple
related tasks that share structure but differ in specific parameters.
"""

import gymnasium as gym
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class TaskDistribution:
    """Base class for task distributions."""
    
    def __init__(self, env_name: str, task_params: List[Any]):
        """
        Args:
            env_name: Base Gymnasium environment name
            task_params: List of task-specific parameters
        """
        self.env_name = env_name
        self.task_params = task_params
        self.num_tasks = len(task_params)
    
    def sample_task(self, task_id: Optional[int] = None):
        """Sample a task from the distribution."""
        if task_id is None:
            task_id = np.random.randint(0, self.num_tasks)
        return self.create_task(task_id)
    
    def create_task(self, task_id: int):
        """Create a specific task by ID."""
        raise NotImplementedError
    
    def get_all_tasks(self):
        """Get all tasks in the distribution."""
        return [self.create_task(i) for i in range(self.num_tasks)]


class VelocityTask(gym.Wrapper):
    """Wrapper that modifies reward based on target velocity."""
    
    def __init__(self, env, target_velocity: float):
        super().__init__(env)
        self.target_velocity = target_velocity
        self._episode_reward = 0
        self._episode_steps = 0
    
    def reset(self, **kwargs):
        self._episode_reward = 0
        self._episode_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get velocity from info or compute from observation
        if 'x_velocity' in info:
            velocity = info['x_velocity']
        else:
            # For environments that don't provide velocity in info
            # Estimate from observation (position change)
            velocity = obs[8] if len(obs) > 8 else 0.0
        
        # Modify reward to encourage target velocity
        velocity_reward = -abs(velocity - self.target_velocity)
        
        # Combine with original reward (weighted)
        modified_reward = 0.5 * reward + 0.5 * velocity_reward
        
        self._episode_reward += modified_reward
        self._episode_steps += 1
        
        info['target_velocity'] = self.target_velocity
        info['actual_velocity'] = velocity
        info['velocity_error'] = abs(velocity - self.target_velocity)
        
        return obs, modified_reward, terminated, truncated, info


class DirectionTask(gym.Wrapper):
    """Wrapper that modifies reward based on target direction."""
    
    def __init__(self, env, target_direction: float):
        """
        Args:
            target_direction: Target direction in radians [0, 2π]
        """
        super().__init__(env)
        self.target_direction = target_direction
        self._initial_pos = None
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Store initial position
        if hasattr(self.env.unwrapped, 'data'):
            self._initial_pos = self.env.unwrapped.data.qpos[:2].copy()
        else:
            self._initial_pos = np.zeros(2)
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current position
        if hasattr(self.env.unwrapped, 'data'):
            current_pos = self.env.unwrapped.data.qpos[:2]
        else:
            current_pos = obs[:2]
        
        # Compute movement direction
        displacement = current_pos - self._initial_pos
        if np.linalg.norm(displacement) > 0.01:
            actual_direction = np.arctan2(displacement[1], displacement[0])
            
            # Compute direction error (circular distance)
            direction_error = np.abs(np.arctan2(
                np.sin(actual_direction - self.target_direction),
                np.cos(actual_direction - self.target_direction)
            ))
            
            direction_reward = -direction_error
        else:
            direction_reward = 0.0
            direction_error = 0.0
        
        # Combine with original reward
        modified_reward = 0.5 * reward + 0.5 * direction_reward
        
        info['target_direction'] = self.target_direction
        info['direction_error'] = direction_error
        
        return obs, modified_reward, terminated, truncated, info


class GravityTask(gym.Wrapper):
    """Wrapper that modifies environment gravity."""
    
    def __init__(self, env, gravity_multiplier: float):
        super().__init__(env)
        self.gravity_multiplier = gravity_multiplier
        self._original_gravity = None
    
    def reset(self, **kwargs):
        # Modify gravity
        if hasattr(self.env.unwrapped, 'model'):
            if self._original_gravity is None:
                self._original_gravity = self.env.unwrapped.model.opt.gravity.copy()
            self.env.unwrapped.model.opt.gravity[:] = (
                self._original_gravity * self.gravity_multiplier
            )
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['gravity_multiplier'] = self.gravity_multiplier
        return obs, reward, terminated, truncated, info


class MassTask(gym.Wrapper):
    """Wrapper that modifies body mass."""
    
    def __init__(self, env, mass_multiplier: float):
        super().__init__(env)
        self.mass_multiplier = mass_multiplier
        self._original_mass = None
    
    def reset(self, **kwargs):
        # Modify body mass
        if hasattr(self.env.unwrapped, 'model'):
            if self._original_mass is None:
                self._original_mass = self.env.unwrapped.model.body_mass.copy()
            self.env.unwrapped.model.body_mass[:] = (
                self._original_mass * self.mass_multiplier
            )
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['mass_multiplier'] = self.mass_multiplier
        return obs, reward, terminated, truncated, info


# ============================================================================
# Task Distribution Implementations
# ============================================================================

class HalfCheetahVelDistribution(TaskDistribution):
    """HalfCheetah with different target velocities."""
    
    def __init__(self, velocities: Optional[List[float]] = None):
        if velocities is None:
            velocities = [0.5, 1.0, 1.5, 2.0, 2.5]
        super().__init__('HalfCheetah-v4', velocities)
    
    def create_task(self, task_id: int):
        env = gym.make(self.env_name)
        return VelocityTask(env, self.task_params[task_id])


class AntVelDistribution(TaskDistribution):
    """Ant with different target velocities."""
    
    def __init__(self, velocities: Optional[List[float]] = None):
        if velocities is None:
            velocities = [0.3, 0.6, 0.9, 1.2, 1.5]
        super().__init__('Ant-v4', velocities)
    
    def create_task(self, task_id: int):
        env = gym.make(self.env_name)
        return VelocityTask(env, self.task_params[task_id])


class AntDirDistribution(TaskDistribution):
    """Ant with different target directions."""
    
    def __init__(self, directions: Optional[List[float]] = None):
        if directions is None:
            # Directions in radians: 0°, 45°, 90°, 135°, 180°
            directions = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        super().__init__('Ant-v4', directions)
    
    def create_task(self, task_id: int):
        env = gym.make(self.env_name)
        return DirectionTask(env, self.task_params[task_id])


class Walker2dVelDistribution(TaskDistribution):
    """Walker2d with different target velocities."""
    
    def __init__(self, velocities: Optional[List[float]] = None):
        if velocities is None:
            velocities = [0.5, 1.0, 1.5, 2.0, 2.5]
        super().__init__('Walker2d-v4', velocities)
    
    def create_task(self, task_id: int):
        env = gym.make(self.env_name)
        return VelocityTask(env, self.task_params[task_id])


class HalfCheetahGravityDistribution(TaskDistribution):
    """HalfCheetah with different gravity values."""
    
    def __init__(self, gravity_multipliers: Optional[List[float]] = None):
        if gravity_multipliers is None:
            gravity_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
        super().__init__('HalfCheetah-v4', gravity_multipliers)
    
    def create_task(self, task_id: int):
        env = gym.make(self.env_name)
        return GravityTask(env, self.task_params[task_id])


class AntMassDistribution(TaskDistribution):
    """Ant with different body mass multipliers."""
    
    def __init__(self, mass_multipliers: Optional[List[float]] = None):
        if mass_multipliers is None:
            mass_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
        super().__init__('Ant-v4', mass_multipliers)
    
    def create_task(self, task_id: int):
        env = gym.make(self.env_name)
        return MassTask(env, self.task_params[task_id])


# ============================================================================
# Registry
# ============================================================================

TASK_DISTRIBUTIONS = {
    'halfcheetah-vel': HalfCheetahVelDistribution,
    'ant-vel': AntVelDistribution,
    'ant-dir': AntDirDistribution,
    'walker2d-vel': Walker2dVelDistribution,
    'halfcheetah-gravity': HalfCheetahGravityDistribution,
    'ant-mass': AntMassDistribution,
}


def get_task_distribution(name: str, **kwargs) -> TaskDistribution:
    """Get a task distribution by name."""
    if name not in TASK_DISTRIBUTIONS:
        raise ValueError(f"Unknown task distribution: {name}. "
                        f"Available: {list(TASK_DISTRIBUTIONS.keys())}")
    return TASK_DISTRIBUTIONS[name](**kwargs)


def list_task_distributions() -> List[str]:
    """List all available task distributions."""
    return list(TASK_DISTRIBUTIONS.keys())


# ============================================================================
# Utility Functions
# ============================================================================

def get_env_info(env) -> Dict[str, Any]:
    """Get environment information."""
    obs_space = env.observation_space
    action_space = env.action_space
    
    return {
        'obs_dim': obs_space.shape[0] if hasattr(obs_space, 'shape') else obs_space.n,
        'action_dim': action_space.shape[0] if hasattr(action_space, 'shape') else action_space.n,
        'obs_space': obs_space,
        'action_space': action_space,
        'is_discrete': isinstance(action_space, gym.spaces.Discrete),
    }


def test_task_distribution(distribution_name: str, num_episodes: int = 3):
    """Test a task distribution."""
    print(f"\n{'='*60}")
    print(f"Testing: {distribution_name}")
    print(f"{'='*60}")
    
    dist = get_task_distribution(distribution_name)
    print(f"Number of tasks: {dist.num_tasks}")
    print(f"Task parameters: {dist.task_params}")
    
    # Test first task
    task = dist.sample_task(task_id=0)
    env_info = get_env_info(task)
    
    print(f"\nEnvironment Info:")
    print(f"  Observation dim: {env_info['obs_dim']}")
    print(f"  Action dim: {env_info['action_dim']}")
    print(f"  Discrete actions: {env_info['is_discrete']}")
    
    # Run a few episodes
    print(f"\nRunning {num_episodes} test episodes...")
    for ep in range(num_episodes):
        obs, info = task.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:
            action = task.action_space.sample()
            obs, reward, terminated, truncated, info = task.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        print(f"  Episode {ep+1}: Reward={episode_reward:.2f}, Steps={steps}")
    
    task.close()
    print(f"✓ {distribution_name} test completed")


if __name__ == "__main__":
    print("SSM-MetaRL Task Distributions")
    print("="*60)
    print("\nAvailable distributions:")
    for name in list_task_distributions():
        print(f"  - {name}")
    
    # Test each distribution
    print("\n" + "="*60)
    print("Running tests...")
    print("="*60)
    
    for dist_name in list_task_distributions():
        try:
            test_task_distribution(dist_name, num_episodes=2)
        except Exception as e:
            print(f"✗ Error testing {dist_name}: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

