"""
Adaptation module for test-time adaptation strategies.

This module provides two adaptation strategies:
1. StandardAdapter: Basic adaptation using only current task data
2. HybridAdapter: Enhanced adaptation with experience replay

Legacy import support:
- Adapter (alias for StandardAdapter for backward compatibility)
"""

from .standard_adapter import StandardAdapter, StandardAdaptationConfig
from .hybrid_adapter import HybridAdapter, HybridAdaptationConfig

# Backward compatibility
Adapter = StandardAdapter

__all__ = [
    'StandardAdapter',
    'StandardAdaptationConfig',
    'HybridAdapter',
    'HybridAdaptationConfig',
    'Adapter',  # Legacy alias
]

