# SSM-MetaRL-Unified

**State Space Model + Meta-Reinforcement Learning with Test-Time Adaptation**

A complete implementation combining State Space Models (SSM) with Meta-RL (MAML) for fast adaptation to new tasks.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/stargatek1/SSM-MetaRL-Unified)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- ✅ **State Space Model (SSM)** architecture for temporal modeling
- ✅ **Meta-Learning with MAML** for fast adaptation
- ✅ **Standard test-time adaptation** using current task data
- ✅ **Hybrid adaptation** with experience replay buffer
- ✅ **Gradio web interface** for interactive experimentation
- ✅ **Original research implementation** maintained

## 🎯 What is SSM-MetaRL?

This project combines two powerful concepts:

1. **State Space Models (SSM)**: Efficient sequential modeling with hidden state
2. **Meta-Reinforcement Learning (MAML)**: Learning to learn for fast adaptation

The result is an agent that can:
- Learn from multiple tasks during meta-training
- Quickly adapt to new tasks with minimal data
- Leverage past experiences for better adaptation

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/SSM-MetaRL-Unified.git
cd SSM-MetaRL-Unified

# Install dependencies
pip install torch gymnasium numpy gradio
```

## 🏃 Quick Start

### Web Interface (Recommended)

Launch the Gradio interface:

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

### Command Line

Meta-training with MAML:

```bash
python main.py \
    --mode meta_rl \
    --env_name CartPole-v1 \
    --num_epochs 100 \
    --tasks_per_epoch 5 \
    --inner_lr 0.01 \
    --outer_lr 0.001
```

Policy gradient training:

```bash
python main.py \
    --mode policy_gradient \
    --env_name CartPole-v1 \
    --num_episodes 200
```

## 🌐 Try it Online

**[🤗 Hugging Face Space](https://huggingface.co/spaces/stargatek1/SSM-MetaRL-Unified)**

No installation required! Try the model directly in your browser.

## 🏗️ Architecture

### State Space Model (SSM)

```
Input (observation) → SSM Block → Output (actions/predictions)
                         ↓
                   Hidden State (recurrent)
```

**Components:**
- State transition network (A matrix)
- Input projection (B matrix)
- Output network (C matrix)
- Feedthrough connection (D matrix)

### Meta-Learning (MAML)

**Inner Loop (Task Adaptation):**
1. Collect support set from task
2. Perform gradient steps on support set
3. Obtain adapted parameters

**Outer Loop (Meta-Update):**
1. Evaluate adapted parameters on query set
2. Compute meta-loss
3. Update meta-parameters

### Test-Time Adaptation

**Standard Mode:**
- Uses only current task observations
- Simple and fast baseline

**Hybrid Mode:**
- Combines current observations + experience replay
- More robust adaptation
- Original research contribution

## 📊 Performance

**CartPole-v1:**
- Meta-training: 100 epochs, 5 tasks/epoch
- Zero-shot: 20-40 reward
- After adaptation: 40-80 reward
- Clear meta-learning benefit

## 📁 Project Structure

```
SSM-MetaRL-Unified/
├── core/
│   ├── ssm.py              # State Space Model
│   └── __init__.py
├── meta_rl/
│   ├── meta_maml.py        # MAML algorithm
│   └── __init__.py
├── adaptation/
│   ├── standard_adapter.py # Standard adaptation
│   ├── hybrid_adapter.py   # Hybrid adaptation
│   └── __init__.py
├── experience/
│   ├── experience_buffer.py # Experience replay
│   └── __init__.py
├── env_runner/
│   ├── environment.py      # Gym wrapper
│   └── __init__.py
├── app.py                  # Gradio interface
├── main.py                 # Training script
└── README.md
```

## 🔬 Key Concepts

### Meta-Learning
Learning to learn - finding good initialization for fast adaptation.

### MAML
Model-Agnostic Meta-Learning through gradient descent.

### State Space Models
Efficient sequential models with hidden state for temporal dependencies.

### Experience Replay
Storing and reusing past experiences for better learning.

## 🎓 Research Background

Based on:
- [MAML Paper](https://arxiv.org/abs/1703.03400)
- [Meta-RL Survey](https://arxiv.org/abs/1910.03193)
- [State Space Models](https://arxiv.org/abs/2111.00396)

## 📝 License

MIT License

## 🙏 Acknowledgments

- MAML algorithm by Finn et al.
- Gymnasium for RL environments
- Gradio for web interface

## 🔗 Links

- **Hugging Face**: https://huggingface.co/spaces/stargatek1/SSM-MetaRL-Unified
- **GitHub**: https://github.com/sunghunkwag/SSM-MetaRL-Unified

---

**Made with ❤️ for the Meta-RL community**

