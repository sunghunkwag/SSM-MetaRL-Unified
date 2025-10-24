#!/usr/bin/env python3
"""
setup.py for SSM-MetaRL-Unified

This file provides backward compatibility and ensures the package
works correctly with older build tools.
"""

import sys
from pathlib import Path

# For projects using pyproject.toml, we just point to setuptools
if __name__ == "__main__":
    # Read version from __init__.py
    init_file = Path(__file__).parent / "__init__.py"
    version = "1.0.0"
    
    if init_file.exists():
        with open(init_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    version = line.split('=')[1].strip().strip('"').strip("'")
                    break
    
    print(f"SSM-MetaRL-Unified v{version}")
    print("This project uses pyproject.toml for configuration.")
    print("Please use 'pip install -e .' or 'python -m build' for installation.")
    
    # For legacy compatibility, we can fall back to setuptools
    try:
        from setuptools import setup
        setup(
            name="ssm-metarl-unified",
            version=version,
            description="Unified State Space Models for Meta-RL with Experience-Augmented Test-Time Adaptation",
        )
    except ImportError:
        print("Error: setuptools not available. Please install it first.")
        sys.exit(1)