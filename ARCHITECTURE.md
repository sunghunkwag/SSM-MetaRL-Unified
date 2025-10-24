# SSM-MetaRL-Unified: Architecture Documentation

## Overview

This document describes the unified architecture that integrates two powerful approaches to meta-reinforcement learning:

1. **SSM-MetaRL-TestCompute**: A robust baseline with State Space Models and test-time adaptation
2. **EAML-SSM**: An innovative experience-augmented meta-learning approach

## Integration Strategy

The integration follows a **5-phase architecture design** that preserves the strengths of both original repositories while creating a seamless unified system.

---

## Phase 1: Experience Module Integration

### Objective
Add the experience replay capability from EAML-SSM to the SSM-MetaRL-TestCompute codebase.

### Implementation
- **Source**: `EAML-SSM/experience/` directory
- **Destination**: `SSM-MetaRL-Unified/experience/`
- **Files Added**:
  - `__init__.py`: Module initialization
  - `experience_buffer.py`: ExperienceBuffer implementation

### ExperienceBuffer Design

The `ExperienceBuffer` is a PyTorch-based circular memory buffer that stores past experiences as `(observation, target)` tensor pairs.

**Key Features:**
- Circular buffer with automatic oldest-item eviction
- PyTorch tensor storage for GPU compatibility
- Random batch sampling for experience replay
- Device-aware tensor management

**API:**
```python
class ExperienceBuffer:
    def __init__(self, max_size: int, device: str)
    def add(self, observations: Tensor, targets: Tensor) -> None
    def get_batch(self, batch_size: int) -> Optional[Tuple[Tensor, Tensor]]
    def __len__(self) -> int
```

---

## Phase 2: Adaptation Module Refactoring

### Objective
Separate the overlapping adapter implementations into distinct classes for standard and hybrid adaptation.

### Implementation

#### 2.1 StandardAdapter (Baseline)

**Source**: `SSM-MetaRL-TestCompute/adaptation/test_time_adaptation.py`

**Purpose**: Performs classic test-time adaptation using only current task data.

**Key Characteristics:**
- No experience replay
- Single-source loss (current data only)
- Baseline for comparison

**API:**
```python
class StandardAdapter:
    def __init__(self, model, config, device)
    def update_step(self, x, y, hidden_state) -> Tuple[float, int]
```

#### 2.2 HybridAdapter (Experience-Augmented)

**Source**: `EAML-SSM/adaptation/test_time_adaptation.py`

**Purpose**: Performs experience-augmented adaptation using a hybrid loss function.

**Key Characteristics:**
- Requires ExperienceBuffer
- Dual-source loss (current + past experiences)
- Weighted combination of losses

**Hybrid Loss Formula:**
```
total_loss = loss_current + α * loss_experience

where:
  loss_current = MSE(model(x_current), y_current)
  loss_experience = MSE(model(x_experience), y_experience)
  α = experience_weight (configurable)
```

**API:**
```python
class HybridAdapter:
    def __init__(self, model, config, experience_buffer, device)
    def update_step(self, x_current, y_current, hidden_state_current) -> Tuple[float, int]
```

#### 2.3 Module Initialization

**File**: `adaptation/__init__.py`

Exports both adapters for easy access:
```python
from .standard_adapter import StandardAdapter, StandardAdaptationConfig
from .hybrid_adapter import HybridAdapter, HybridAdaptationConfig
```

---

## Phase 3: Package Configuration Update

### Objective
Register the new `experience` module in the package configuration.

### Implementation

**File**: `pyproject.toml`

**Changes:**
```toml
[project]
name = "ssm-metarl-unified"
version = "1.0.0"
description = "Unified State Space Models for Meta-RL with Experience-Augmented Test-Time Adaptation"

[tool.setuptools.packages.find]
include = [
    "core", "core.*",
    "meta_rl", "meta_rl.*",
    "env_runner", "env_runner.*",
    "adaptation", "adaptation.*",
    "experience", "experience.*"  # NEW
]
```

---

## Phase 4: Main Script Enhancement

### Objective
Enable dual adaptation mode support in the main training script.

### Implementation

**File**: `main.py`

#### 4.1 New Imports
```python
from adaptation import StandardAdapter, StandardAdaptationConfig
from adaptation import HybridAdapter, HybridAdaptationConfig
from experience.experience_buffer import ExperienceBuffer
```

#### 4.2 Enhanced `collect_data` Function

**New Parameter**: `experience_buffer` (optional)

**Behavior**: When provided, automatically populates the buffer during data collection.

```python
def collect_data(env, policy_model, ..., experience_buffer=None):
    # ... collect trajectory ...
    
    if experience_buffer is not None:
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
        experience_buffer.add(obs_t.unsqueeze(0), next_obs_t.unsqueeze(0))
```

#### 4.3 Enhanced `train_meta` Function

**New Parameter**: `experience_buffer` (optional)

**Behavior**: Passes buffer to `collect_data` and logs buffer size during training.

#### 4.4 Enhanced `test_time_adapt` Function

**New Parameters**:
- `adaptation_mode`: `'standard'` or `'hybrid'`
- `experience_buffer`: Required for hybrid mode

**Behavior**: Creates appropriate adapter based on mode.

```python
if adaptation_mode == 'hybrid':
    config = HybridAdaptationConfig(...)
    adapter = HybridAdapter(model, config, experience_buffer, device)
else:
    config = StandardAdaptationConfig(...)
    adapter = StandardAdapter(model, config, device)
```

#### 4.5 New Command-Line Arguments

```python
parser.add_argument('--adaptation_mode', choices=['standard', 'hybrid'], default='standard')
parser.add_argument('--buffer_size', type=int, default=10000)
parser.add_argument('--experience_batch_size', type=int, default=32)
parser.add_argument('--experience_weight', type=float, default=0.1)
```

---

## Phase 5: SOTA Benchmark Enhancement

### Objective
Enable dual adaptation mode support in the serious benchmark suite.

### Implementation

**File**: `experiments/serious_benchmark.py`

#### 5.1 BenchmarkRunner Initialization

**New Logic**: Initialize ExperienceBuffer if hybrid mode is selected.

```python
def __init__(self, task_dist_name, method_name, config):
    # ... existing initialization ...
    
    self.experience_buffer = None
    if config.get('adaptation_mode', 'standard') == 'hybrid':
        self.experience_buffer = ExperienceBuffer(
            max_size=config.get('buffer_size', 10000),
            device=str(self.device)
        )
```

#### 5.2 Enhanced `collect_episode_data`

**New Parameter**: `populate_buffer` (boolean)

**Behavior**: Conditionally adds data to buffer during collection.

```python
def collect_episode_data(self, env, max_steps, populate_buffer=False):
    # ... collect data ...
    
    if populate_buffer and self.experience_buffer is not None:
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        self.experience_buffer.add(obs_t.unsqueeze(0), next_obs_t.unsqueeze(0))
```

#### 5.3 Enhanced `meta_train_step`

**Behavior**: Populates buffer during training by passing `populate_buffer=True`.

#### 5.4 Enhanced `meta_test`

**Behavior**: Creates appropriate adapter based on `adaptation_mode` configuration.

```python
adaptation_mode = self.config.get('adaptation_mode', 'standard')

if adaptation_mode == 'hybrid':
    config = HybridAdaptationConfig(...)
    adapter = HybridAdapter(model, config, self.experience_buffer, device)
else:
    config = StandardAdaptationConfig(...)
    adapter = StandardAdapter(model, config, device)
```

**Important**: Uses `populate_buffer=False` during testing to prevent buffer contamination.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SSM-MetaRL-Unified                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   core/      │      │  meta_rl/    │                    │
│  │   - ssm.py   │      │  - meta_maml │                    │
│  └──────────────┘      └──────────────┘                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              adaptation/                              │  │
│  │  ┌─────────────────┐    ┌─────────────────┐         │  │
│  │  │ StandardAdapter │    │  HybridAdapter  │         │  │
│  │  │   (baseline)    │    │ (experience-    │         │  │
│  │  │                 │    │  augmented)     │         │  │
│  │  └─────────────────┘    └────────┬────────┘         │  │
│  │                                   │                   │  │
│  └───────────────────────────────────┼───────────────────┘  │
│                                      │                       │
│  ┌───────────────────────────────────▼───────────────────┐  │
│  │              experience/                              │  │
│  │           ExperienceBuffer                            │  │
│  │  - Circular buffer for past experiences              │  │
│  │  - Random batch sampling                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ env_runner/  │      │ experiments/ │                    │
│  │ - environment│      │ - benchmarks │                    │
│  └──────────────┘      └──────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Adapter Separation

**Rationale**: Separating standard and hybrid adapters provides:
- Clear code organization
- Easy mode switching
- Fair performance comparison
- Minimal code duplication

### 2. Optional Buffer Parameter

**Rationale**: Making the buffer optional allows:
- Backward compatibility with standard mode
- Flexible buffer initialization
- Clear dependency management

### 3. Hybrid Loss Weighting

**Rationale**: The weighted combination allows:
- Tunable balance between current and past data
- Adaptation to different task characteristics
- Empirical optimization of the experience weight

### 4. Buffer Population Control

**Rationale**: The `populate_buffer` flag prevents:
- Test data leakage into training buffer
- Contamination of experience distribution
- Unfair benchmark comparisons

---

## Testing Strategy

### Integration Tests

**File**: `test_integration.py`

**Coverage**:
1. ExperienceBuffer functionality
2. StandardAdapter correctness
3. HybridAdapter with experience replay
4. End-to-end workflow for both modes

### Test Results

```
======================================================================
TEST SUMMARY
======================================================================
ExperienceBuffer               ✓ PASSED
Standard Mode                  ✓ PASSED
Hybrid Mode                    ✓ PASSED
======================================================================
ALL TESTS PASSED ✓
======================================================================
```

---

## Usage Examples

### Standard Mode

```bash
python main.py \
    --env_name CartPole-v1 \
    --adaptation_mode standard \
    --num_epochs 20
```

### Hybrid Mode

```bash
python main.py \
    --env_name CartPole-v1 \
    --adaptation_mode hybrid \
    --buffer_size 5000 \
    --experience_weight 0.15 \
    --num_epochs 20
```

### Benchmark Comparison

```bash
# Standard
python experiments/serious_benchmark.py \
    --task halfcheetah-vel \
    --method ssm \
    --adaptation_mode standard \
    --epochs 50

# Hybrid
python experiments/serious_benchmark.py \
    --task halfcheetah-vel \
    --method ssm \
    --adaptation_mode hybrid \
    --buffer_size 20000 \
    --experience_weight 0.1 \
    --epochs 50
```

---

## Future Extensions

### Potential Enhancements

1. **Prioritized Experience Replay**: Weight experiences by TD-error or surprise
2. **Dynamic Buffer Sizing**: Automatically adjust buffer size based on task complexity
3. **Multi-Task Buffers**: Separate buffers for different task distributions
4. **Episodic Memory**: Store complete episodes rather than individual transitions
5. **Meta-Learning the Experience Weight**: Learn optimal α through meta-gradient

---

## References

- **State Space Models**: Efficient sequence modeling with linear complexity
- **MAML**: Model-Agnostic Meta-Learning for fast adaptation
- **Experience Replay**: Stabilizing RL through past experience reuse
- **Test-Time Adaptation**: Online learning during deployment

---

## Conclusion

The SSM-MetaRL-Unified architecture successfully integrates two complementary approaches to meta-reinforcement learning. The modular design allows researchers to easily compare standard and experience-augmented adaptation, while the comprehensive testing ensures reliability and correctness.

The unified framework is ready for:
- Research experimentation
- SOTA benchmark evaluation
- Extension with new adaptation strategies
- Deployment in real-world applications

