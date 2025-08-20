# SMAC Experiment Architecture: AMAL vs MAPPO

## 🎯 Goal
Compare AMAL (Asymmetric Multi-Agent Learning) with MAPPO baseline on SMAC scenarios to validate the paper's claims of 40% better sample efficiency.

## 📊 Expected Performance
Based on the paper (Table 1):
- **MAPPO on SMAC**: 75.4% win rate
- **AMAL target**: 81.6% win rate (~8% improvement)
- **Sample efficiency**: 40% fewer samples to reach 90% performance

## 🏗️ Architecture Overview

```
smac_experiments/
├── core/
│   ├── __init__.py
│   ├── base_algorithm.py      # Abstract base class
│   ├── amal_agent.py          # AMAL implementation
│   └── mappo_baseline.py      # MAPPO implementation
├── environments/
│   ├── __init__.py
│   └── smac_wrapper.py        # SMAC environment wrapper
├── networks/
│   ├── __init__.py
│   ├── world_model.py         # AMAL world model
│   ├── policy_net.py          # Shared policy architecture
│   └── critic_net.py          # Shared critic architecture
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py       # Asymmetric replay buffer
│   ├── mi_estimator.py        # Mutual information estimator
│   └── logger.py              # Experiment logging
├── configs/
│   ├── amal_config.yaml       # AMAL hyperparameters
│   ├── mappo_config.yaml      # MAPPO hyperparameters
│   └── smac_scenarios.yaml    # SMAC scenario configs
├── run_experiment.py           # Main experiment runner
└── plot_results.py            # Result visualization
```

## 🔌 Core Interfaces

### 1. Base Algorithm Interface
```python
from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Tuple, Optional

class BaseMAAlgorithm(ABC):
    """Base class for multi-agent algorithms"""
    
    @abstractmethod
    def __init__(self, config: dict):
        """Initialize algorithm with config"""
        pass
    
    @abstractmethod
    def select_actions(
        self, 
        obs: Dict[int, torch.Tensor],
        available_actions: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[int, torch.Tensor]:
        """Select actions for all agents"""
        pass
    
    @abstractmethod
    def update(self, batch: Dict) -> Dict[str, float]:
        """Update algorithm parameters"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model checkpoint"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model checkpoint"""
        pass
```

### 2. Replay Buffer Interface
```python
class AsymmetricReplayBuffer:
    """Replay buffer with asymmetric data handling"""
    
    def add_primary(self, transition: Dict):
        """Add primary agent experience"""
        
    def add_auxiliary(self, agent_id: int, transition: Dict):
        """Add auxiliary agent experience"""
        
    def sample_primary(self, batch_size: int) -> Dict:
        """Sample ONLY from primary buffer"""
        
    def sample_mixed(self, batch_size: int) -> Dict:
        """Sample from all buffers (for baselines)"""
```

### 3. World Model Interface
```python
class WorldModel(nn.Module):
    """World model for AMAL"""
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next observation and reward"""
        
    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute prediction loss"""
```

## 📋 SMAC Scenarios for Testing

### Easy (2-3 agents)
- **2s3z**: 2 Stalkers vs 3 Zealots
- **3m**: 3 Marines

### Medium (5-10 agents)
- **5m_vs_6m**: 5 Marines vs 6 Marines
- **8m**: 8 Marines
- **2s_vs_1sc**: 2 Stalkers vs 1 Spine Crawler

### Hard (10+ agents)
- **3s5z**: 3 Stalkers and 5 Zealots
- **MMM**: 1 Medivac, 2 Marauders, 7 Marines
- **27m_vs_30m**: 27 Marines vs 30 Marines

### Super Hard
- **corridor**: 6 Zealots vs 24 Zerglings
- **6h_vs_8z**: 6 Hydralisks vs 8 Zealots

## 🤖 Recommended Model for Implementation

**Use Claude 3.5 Sonnet** (cheaper than Claude 4.1/Opus) with these settings:
- Temperature: 0.2 (for consistent code)
- Max tokens: 4000
- Stop sequences: ["```\n\n", "# End"]

## 📝 Implementation Prompts (Step by Step)

### Phase 1: Core Infrastructure
```
PROMPT 1: Create base algorithm interface
"Create a BaseMAAlgorithm abstract class for SMAC experiments with methods for:
- select_actions (handle multiple agents with available actions mask)
- update (return loss dict)
- save/load checkpoints
Include proper typing hints and docstrings."
```

### Phase 2: AMAL Implementation
```
PROMPT 2: Implement AMAL core logic
"Implement AMAL agent class inheriting from BaseMAAlgorithm with:
1. Asymmetric update rule (world model uses ONLY primary data)
2. Information-seeking objective with MI estimation
3. Auxiliary agent management
Follow the mathematical formulation from the paper section 4."
```

### Phase 3: MAPPO Baseline
```
PROMPT 3: Implement MAPPO baseline
"Implement MAPPO baseline for SMAC following Yu et al. 2022:
- Centralized critic with global state
- PPO clipping for policy updates
- Value normalization
- Proper credit assignment
Use standard hyperparameters from the paper."
```

### Phase 4: SMAC Integration
```
PROMPT 4: Create SMAC wrapper
"Create SMAC environment wrapper that:
- Handles multiple agents observation/action spaces
- Manages available actions masking
- Provides both local obs and global state
- Tracks episode statistics
Compatible with both AMAL and MAPPO."
```

### Phase 5: Training Loop
```
PROMPT 5: Main training script
"Create training script that:
1. Runs both AMAL and MAPPO on same scenarios
2. Logs metrics (win rate, sample efficiency, training time)
3. Saves checkpoints every 100 episodes
4. Handles early stopping
5. Supports parallel environments"
```

## 📊 Evaluation Metrics

```python
metrics = {
    'win_rate': [],           # Primary metric
    'episode_return': [],     # Total reward
    'battle_won': [],         # Binary win/loss
    'dead_allies': [],        # Units lost
    'dead_enemies': [],       # Enemies killed
    'sample_efficiency': [],  # Episodes to 90% performance
    'training_time': [],      # Wall clock time
    'gpu_memory': []         # Memory usage
}
```

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install pymarl2 torch wandb

# Run easy scenario
python run_experiment.py --algo amal --scenario 3m --episodes 1000

# Run comparison
python run_experiment.py --compare --scenarios 3m,8m,MMM --episodes 5000

# Plot results
python plot_results.py --exp_dir results/
```

## 💡 Success Probability Analysis

### High Confidence (80-90%):
- **Information isolation**: Mechanical property, will work as designed
- **Basic functionality**: Both algorithms will run and train

### Medium Confidence (50-70%):
- **8% performance gain**: Achievable with proper hyperparameter tuning
- **Sample efficiency**: Should see improvement, maybe not full 40%

### Challenges & Mitigations:
1. **SMAC complexity**: Start with easy scenarios (3m, 2s3z)
2. **Hyperparameter sensitivity**: Use grid search on key params (λ, num_aux_agents)
3. **Computational cost**: Use smaller networks initially (256 hidden vs 512)
4. **Baseline strength**: MAPPO is already very strong on SMAC

### Realistic Expectations:
- **Week 1**: Get both algorithms running (basic functionality)
- **Week 2**: Achieve competitive performance on easy scenarios
- **Week 3**: Tune for harder scenarios, optimize efficiency
- **Success metric**: Beat MAPPO on 3+ scenarios with better sample efficiency

## 🔧 Optimization Tips

1. **Start small**: 3m scenario with 100 episodes for debugging
2. **Profile first**: Check GPU utilization before scaling
3. **Batch environments**: Use 8-16 parallel SMAC instances
4. **Mixed precision**: Enable for 2x speedup on modern GPUs
5. **Gradient accumulation**: Simulate larger batches on limited memory

## 📈 Expected Timeline

- **Day 1**: Setup environment, implement base classes (2-3 hours)
- **Day 2**: AMAL core implementation (3-4 hours)
- **Day 3**: MAPPO baseline (2-3 hours)
- **Day 4**: SMAC integration and testing (2-3 hours)
- **Day 5**: Hyperparameter tuning (4-5 hours)
- **Day 6-7**: Full experiments and analysis (8-10 hours)

**Total estimated time**: 20-30 hours of active development
**Compute time**: 50-100 GPU hours for full experiments

## 🎯 Key to Success

1. **Start with working baseline**: Get MAPPO running first
2. **Incremental testing**: Verify each component separately
3. **Monitor closely**: Use wandb/tensorboard from the start
4. **Parameter sweep**: Focus on λ (info weight) and num_auxiliary_agents
5. **Fair comparison**: Same network architecture for both algorithms
