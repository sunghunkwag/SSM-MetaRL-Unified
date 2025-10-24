"""SSM-MetaRL-Unified: Experience-Augmented Meta-RL Framework.

A unified research framework combining State Space Models (SSM), Meta-Learning (MAML), 
and Experience-Augmented Test-Time Adaptation for advanced reinforcement learning.
"""

__version__ = "1.0.0"
__author__ = "Manus AI"
__email__ = "ai@manus.im"
__description__ = "Unified State Space Models for Meta-RL with Experience-Augmented Test-Time Adaptation"

# Core imports for easy access
from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation import StandardAdapter, HybridAdapter, StandardAdaptationConfig, HybridAdaptationConfig
from experience.experience_buffer import ExperienceBuffer
from env_runner.environment import Environment

__all__ = [
    'StateSpaceModel',
    'MetaMAML', 
    'StandardAdapter',
    'HybridAdapter',
    'StandardAdaptationConfig',
    'HybridAdaptationConfig',
    'ExperienceBuffer',
    'Environment',
]