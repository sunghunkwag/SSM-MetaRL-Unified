# SSM-MetaRL-Unified: Experience-Augmented Meta-RL

**A unified research framework combining State Space Models (SSM), Meta-Learning (MAML), and Experience-Augmented Test-Time Adaptation for advanced reinforcement learning.**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/sunghunkwag/SSM-MetaRL-Unified)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-supported-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/SSM-MetaRL-Unified/blob/master/demo.ipynb)

---

## 🚀 Overview & Key Innovation

**SSM-MetaRL-Unified** is a cutting-edge framework that integrates two powerful concepts in reinforcement learning:

1.  **`SSM-MetaRL-TestCompute`**: A robust baseline for meta-learning with State Space Models (SSM) and test-time adaptation, proven on high-dimensional SOTA benchmarks.
2.  **`EAML-SSM`**: An innovative approach for **Experience-Augmented Meta-Learning**, which enhances adaptation by replaying past experiences.

This unified repository merges these two projects, allowing researchers to seamlessly switch between **standard adaptation** and **experience-augmented (hybrid) adaptation**. The core innovation is the ability to leverage an `ExperienceBuffer` to regularize and guide adaptation, leading to more robust and sample-efficient learning, especially in sparse or non-stationary environments.

### Core Features

-   **State Space Models (SSM)**: For efficient and powerful temporal dynamics modeling.
-   **Meta-Learning (MAML)**: For rapid adaptation to new tasks.
-   **Dual Adaptation Modes**: 
    -   `standard`: Classic test-time adaptation using only current task data.
    -   `hybrid`: **Experience-augmented adaptation** using a hybrid loss that combines current data with past experiences from a replay buffer.
-   **Modular & Extensible**: Cleanly separated modules for models, meta-learning, adaptation, and experience replay.
-   **SOTA Benchmarks**: Includes high-dimensional MuJoCo tasks to validate performance against baselines like LSTM, GRU, and Transformer.
-   **Docker Support**: Pre-configured container for easy deployment and reproducible experiments.
-   **Colab Integration**: An interactive demo notebook to explore the framework's capabilities without local installation.

---

## 🔧 Unified Architecture

The integration combines the strengths of both original repositories into a single, cohesive structure. The key changes are:

-   **`experience` Module**: The `ExperienceBuffer` from `EAML-SSM` is now a core module, enabling the storage and sampling of past trajectories.
-   **Refactored `adaptation` Module**: The adaptation logic is split into two distinct classes:
    -   `StandardAdapter`: Performs adaptation using only current task data (from `SSM-MetaRL-TestCompute`).
    -   `HybridAdapter`: Implements experience-augmented adaptation by combining current loss with a loss computed on a batch of past experiences (from `EAML-SSM`).
-   **Unified `main.py` and `serious_benchmark.py`**: Both scripts now include an `--adaptation_mode` flag, allowing you to easily switch between `standard` and `hybrid` modes.

### Project Structure

```
/SSM-MetaRL-Unified
├── core/
│   └── ssm.py              # State Space Model implementation
├── meta_rl/
│   └── meta_maml.py        # MetaMAML algorithm
├── adaptation/
│   ├── standard_adapter.py # Standard test-time adaptation
│   └── hybrid_adapter.py   # Experience-augmented adaptation
├── experience/
│   └── experience_buffer.py  # Experience replay buffer
├── experiments/
│   ├── serious_benchmark.py  # SOTA benchmarks (now with hybrid mode)
│   └── ...
├── tests/
│   └── test_integration.py # New integration tests for both modes
├── main.py                   # Main script with dual adaptation modes
├── demo.ipynb                # Interactive Colab demo
├── Dockerfile                # Docker configuration for easy deployment
└── pyproject.toml            # Unified project configuration
```

---

## 🚀 Try it Now in Google Colab!

**No installation required!** Click the badge below to run the interactive demo in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/SSM-MetaRL-Unified/blob/master/demo.ipynb)

The demo notebook includes:
- ✅ Complete setup and installation
- ✅ Meta-training with MetaMAML
- ✅ Standard adaptation mode (baseline)
- ✅ Hybrid adaptation mode (experience-augmented)
- ✅ Side-by-side comparison with visualizations
- ✅ Interactive exploration of the framework

---

## ⚡ Quick Start

### 🐳 Option 1: Docker (Recommended)

**The fastest way to get started!** No complex dependency management required.

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/SSM-MetaRL-Unified.git
cd SSM-MetaRL-Unified

# Build the Docker image
docker build -t ssm-metarl-unified .

# Run standard adaptation mode
docker run --rm ssm-metarl-unified python main.py \
    --env_name CartPole-v1 \
    --adaptation_mode standard \
    --num_epochs 10

# Run hybrid adaptation mode (experience-augmented)
docker run --rm ssm-metarl-unified python main.py \
    --env_name CartPole-v1 \
    --adaptation_mode hybrid \
    --num_epochs 10 \
    --buffer_size 5000 \
    --experience_weight 0.2

# Run benchmarks
docker run --rm ssm-metarl-unified python experiments/serious_benchmark.py \
    --task halfcheetah-vel \
    --method ssm \
    --adaptation_mode hybrid \
    --epochs 50

# Run integration tests
docker run --rm ssm-metarl-unified python test_integration.py
```

### 📦 Option 2: Local Installation

```bash
# Clone the unified repository
git clone https://github.com/sunghunkwag/SSM-MetaRL-Unified.git
cd SSM-MetaRL-Unified

# Install the package in editable mode
pip install -e .

# For development (including testing dependencies)
pip install -e .[dev]
```

### Running the Main Script

The `main.py` script allows you to test both adaptation modes on a simple environment like `CartPole-v1`.

**1. Standard Adaptation (Baseline)**

This mode uses only the data from the current episode for adaptation.

```bash
python main.py --env_name CartPole-v1 --adaptation_mode standard --num_epochs 10
```

**2. Hybrid Adaptation (Experience-Augmented)**

This mode enhances adaptation by replaying past experiences from the buffer. The `--buffer_size` and `--experience_weight` flags control this behavior.

```bash
python main.py --env_name CartPole-v1 --adaptation_mode hybrid --num_epochs 10 --buffer_size 5000 --experience_weight 0.2
```

### Running the Integration Tests

We have included a new test script to verify that all integrated components work correctly.

```bash
# Local installation
python test_integration.py

# Or with Docker
docker run --rm ssm-metarl-unified python test_integration.py
```

This will run a series of tests to confirm:
-   The `ExperienceBuffer` functions correctly.
-   The `StandardAdapter` runs without errors.
-   The `HybridAdapter` runs and utilizes the experience buffer.

---

## 🏆 SOTA Benchmarking

The `serious_benchmark.py` script has been updated to support both adaptation modes, allowing for a direct comparison of their effectiveness on challenging, high-dimensional tasks.

### Prerequisites (Local Installation Only)

To run the MuJoCo-based benchmarks locally, you need to install the necessary dependencies:

```bash
pip install 'gymnasium[mujoco]'
```

**Note**: Docker image already includes all required dependencies!

### Running Benchmarks

**1. Standard Adaptation Benchmark**

```bash
# Local
python experiments/serious_benchmark.py \
    --task halfcheetah-vel \
    --method ssm \
    --adaptation_mode standard \
    --epochs 50

# Docker
docker run --rm ssm-metarl-unified python experiments/serious_benchmark.py \
    --task halfcheetah-vel \
    --method ssm \
    --adaptation_mode standard \
    --epochs 50
```

**2. Hybrid Adaptation Benchmark**

Compare the performance with experience replay enabled.

```bash
# Local
python experiments/serious_benchmark.py \
    --task halfcheetah-vel \
    --method ssm \
    --adaptation_mode hybrid \
    --epochs 50 \
    --buffer_size 20000 \
    --experience_weight 0.1

# Docker
docker run --rm ssm-metarl-unified python experiments/serious_benchmark.py \
    --task halfcheetah-vel \
    --method ssm \
    --adaptation_mode hybrid \
    --epochs 50 \
    --buffer_size 20000 \
    --experience_weight 0.1
```

### Expected Outcome

By comparing the results from these two runs, you can quantify the benefits of experience-augmented adaptation. We expect the `hybrid` mode to show improved sample efficiency and final performance, especially in tasks where learning from scratch during test-time is difficult.

---

## 🔬 Core Components Explained

### `ExperienceBuffer`

-   **Location**: `experience/experience_buffer.py`
-   **Function**: A circular buffer that stores `(observation, target)` pairs from past trajectories.
-   **API**:
    -   `add(observations, targets)`: Adds new experiences to the buffer.
    -   `get_batch(batch_size)`: Samples a random batch of past experiences.

### `HybridAdapter`

-   **Location**: `adaptation/hybrid_adapter.py`
-   **Function**: Performs test-time adaptation using a **hybrid loss**.
-   **Mechanism**: The total loss is a weighted sum of the loss on the **current data** and the loss on a **batch of past experiences** sampled from the `ExperienceBuffer`. 

    ```python
    # Simplified hybrid loss calculation
    loss_current = self.loss_fn(output_current, y_current)
    
    experience_batch = self.experience_buffer.get_batch(...)
    if experience_batch is not None:
        x_exp, y_exp = experience_batch
        output_exp, _ = self.model(x_exp, ...)
        loss_experience = self.loss_fn(output_exp, y_exp)
        
        # Combine losses
        total_loss = loss_current + self.config.experience_weight * loss_experience
    ```

### Command-Line Arguments

This unified framework introduces several new arguments to control the adaptation process:

-   `--adaptation_mode`: `[standard, hybrid]` - Selects the adaptation strategy.
-   `--buffer_size`: `int` - The maximum number of experiences to store in the buffer.
-   `--experience_batch_size`: `int` - The number of past experiences to sample for each hybrid update.
-   `--experience_weight`: `float` - The weight (alpha) applied to the experience-based loss component.

---

## 🐳 Docker Configuration

The included `Dockerfile` provides a complete, reproducible environment:

- **Multi-stage build**: Optimized image size using builder pattern
- **Python 3.9**: Stable Python version with all required packages
- **Pre-installed dependencies**: All ML/RL libraries ready to use
- **Health check**: Container health monitoring
- **Development ready**: Includes all source code and test scripts

### Docker Commands Summary

```bash
# Build image
docker build -t ssm-metarl-unified .

# Basic run (shows help)
docker run --rm ssm-metarl-unified

# Standard mode example
docker run --rm ssm-metarl-unified python main.py --adaptation_mode standard

# Hybrid mode example  
docker run --rm ssm-metarl-unified python main.py --adaptation_mode hybrid

# Interactive mode (for development)
docker run -it --rm ssm-metarl-unified bash
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this unified framework in your research, please consider citing the original repositories:

```bibtex
@software{ssm_metarl_unified,
  author = {Manus AI},
  title = {SSM-MetaRL-Unified: A Framework for Experience-Augmented Meta-RL},
  year = {2025},
  url = {https://github.com/sunghunkwag/SSM-MetaRL-Unified}
}
```