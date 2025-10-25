# SSM-MetaRL-Unified

**State Space Model + Meta-Reinforcement Learning with Test-Time Adaptation and Recursive Self-Improvement**

A complete implementation combining State Space Models (SSM) with Meta-RL (MAML) for fast adaptation to new tasks, featuring cutting-edge **Recursive Self-Improvement (RSI)** capabilities.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/stargatek1/SSM-MetaRL-Unified)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- ✅ **State Space Model (SSM)** architecture for temporal modeling
- ✅ **Meta-Learning with MAML** for fast adaptation
- ✅ **Standard test-time adaptation** using current task data
- ✅ **Hybrid adaptation** with experience replay buffer
- ✅ **Recursive Self-Improvement (RSI)** - NEW! 🧠
  - Autonomous model improvement
  - Multi-metric evaluation (5 dimensions)
  - Architectural evolution
  - Hyperparameter optimization
  - Safety monitoring with rollback
- ✅ **Pre-trained model** ready to use (no training required)
- ✅ **Background daemon** for continuous improvement
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
├── app.py                  # Gradio interface with RSI
├── main.py                 # Training script
├── recursive_self_improvement.py  # RSI implementation
├── rsi_daemon.py           # Background RSI daemon
├── rsi_control.sh          # Daemon control script
├── cartpole_hybrid_real_model.pth  # Pre-trained weights
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

### Recursive Self-Improvement (RSI)
Autonomous system that continuously improves itself through:
- Multi-metric performance evaluation
- Architectural evolution (testing different model structures)
- Hyperparameter optimization
- Safety monitoring with automatic rollback
- Checkpoint management for recovery

## 🧠 Recursive Self-Improvement (RSI)

### What is RSI?

RSI enables the model to **autonomously improve itself** without human intervention. The system:

1. **Evaluates** current performance across 5 metrics
2. **Proposes** architectural and hyperparameter changes
3. **Tests** each proposal independently
4. **Selects** the best improvement
5. **Applies** changes and saves checkpoints

### Quick Start with RSI

#### Interactive Mode (Gradio)

```bash
python app.py
```

1. Tab 0: Load Pre-trained Model
2. Tab 3: Recursive Self-Improvement 🧠
3. Select cycles (1-10) and click "Run RSI"

#### Background Daemon Mode

Run continuous improvement in the background:

```bash
# Start daemon
./rsi_control.sh start

# Check status
./rsi_control.sh status

# View live logs
./rsi_control.sh logs

# Stop daemon
./rsi_control.sh stop
```

See [RSI_DAEMON_README.md](RSI_DAEMON_README.md) for details.

### RSI Features

**Multi-Metric Evaluation:**
- Average Reward
- Adaptation Speed
- Generalization Score
- Meta-Learning Efficiency
- Stability Score

**Architectural Evolution:**
- State dimension optimization (±20%)
- Hidden dimension tuning (±20%)
- Layer configuration changes

**Safety System:**
- Emergency stop after 3 failures
- Performance threshold monitoring
- Automatic rollback on degradation
- Checkpoint system for recovery

### Performance

Typical improvements:
- **Initial**: 9.60 reward
- **After 1 cycle**: 22.00 reward (+129%)
- **Stability**: 95%+
- **Adaptation Speed**: Optimal

See [RSI_DEPLOYMENT.md](RSI_DEPLOYMENT.md) for complete documentation.

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

