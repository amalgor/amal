# SMAC Implementation Prompts for Claude 3.5 Sonnet

## 🎯 Instructions for Using These Prompts

1. **Model**: Use Claude 3.5 Sonnet (faster and cheaper than Opus)
2. **Temperature**: Set to 0.2 for consistent code
3. **Copy each prompt exactly** as written
4. **Review output** before proceeding to next prompt
5. **Test incrementally** - don't wait until the end

---



```
You are implementing the AMAL (Asymmetric Multi-Agent Learning) algorithm for SMAC experiments.
## 📝 PROMPT 1: AMAL Core Implementation
Create the file `core/amal_agent.py` that inherits from BaseMAAlgorithm with these specifications:

1. ASYMMETRIC UPDATE RULE:
   - World model ONLY updates from primary agent data
   - Policy can use auxiliary agent information for exploration
   - Maintain separate replay buffers

2. INFORMATION-SEEKING OBJECTIVE:
   - Add mutual information term I(θ; O) to policy loss
   - Weight by lambda parameter (default 0.3)
   - Use efficient MI estimator with sampling

3. AUXILIARY AGENTS:
   - Maintain N auxiliary agents (default 16)
   - Evolve using CEM with diversity fitness
   - Generate synthetic experiences for exploration

4. KEY METHODS:
   - __init__: Initialize networks, buffers, auxiliary agents
   - select_actions: Action selection for all agents
   - update: Asymmetric parameter updates
   - _update_world_model: Uses ONLY primary buffer
   - _update_policy: Includes MI bonus
   - _evolve_auxiliary: CEM evolution step

Include proper docstrings and type hints. Follow the mathematical formulation from the AMAL paper Section 4.

Network architectures:
- World model: 3-layer MLP (256-256-128)
- Policy: 3-layer MLP (256-256-action_dim)
- Critic: 3-layer MLP (256-256-1)
```

---

## 📝 PROMPT 2: MAPPO Baseline Implementation

```
You are implementing the MAPPO baseline for SMAC experiments based on "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (Yu et al., 2022).

Create the file `core/mappo_baseline.py` that inherits from BaseMAAlgorithm with:

1. ARCHITECTURE:
   - Decentralized actors (one per agent)
   - Centralized critic using global state
   - Shared parameters across agents

2. PPO COMPONENTS:
   - Clipped surrogate objective (clip_param=0.2)
   - Value function loss with clipping
   - Entropy bonus for exploration (coef=0.01)
   - GAE for advantage estimation (lambda=0.95)

3. MAPPO SPECIFICS:
   - Value normalization (running mean/std)
   - Gradient clipping (max_norm=10.0)
   - Mini-batch updates (4 epochs, 4 mini-batches)
   - Action masking for invalid actions

4. KEY METHODS:
   - __init__: Initialize actor-critic networks
   - select_actions: Decentralized action selection
   - update: PPO update with value normalization
   - _compute_returns: GAE advantage computation
   - _ppo_update: Clipped objective optimization

Hyperparameters from the paper:
- Learning rate: 5e-4 (with linear decay)
- Gamma: 0.99
- Batch size: 32 episodes
- Hidden dims: [256, 256]
```

---

## 📝 PROMPT 3: SMAC Environment Wrapper

```
Create the file `environments/smac_wrapper.py` that wraps SMAC for our experiments:

1. ENVIRONMENT INTERFACE:
   - Handle multiple SMAC scenarios
   - Convert SMAC obs/actions to our format
   - Manage available actions masking
   - Track episode statistics

2. KEY FEATURES:
   - Support both local obs and global state
   - Handle variable number of agents
   - Proper reward shaping (win bonus)
   - Episode info tracking

3. CLASS STRUCTURE:
```python
class SMACWrapper:
    def __init__(self, scenario_name: str, seed: int = 0):
        # Initialize SMAC environment
        
    def reset(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        # Return observations and global state
        
    def step(self, actions: Dict[int, int]) -> Tuple[...]:
        # Execute actions, return next_obs, rewards, done, info
        
    def get_stats(self) -> Dict[str, float]:
        # Return episode statistics
```

4. SUPPORTED SCENARIOS:
   - Easy: 3m, 2s3z
   - Medium: 8m, 5m_vs_6m, 2s_vs_1sc
   - Hard: MMM, 3s5z
   - Super Hard: corridor, 6h_vs_8z

Include conversion between SMAC's observation format and our algorithm's expected format.
```

---

## 📝 PROMPT 4: Asymmetric Replay Buffer

```
Create the file `utils/replay_buffer.py` implementing the asymmetric buffer for AMAL:

1. BUFFER STRUCTURE:
   - Primary buffer: Only real environment data
   - Auxiliary buffers: One per auxiliary agent
   - Strict separation for world model training

2. CLASS IMPLEMENTATION:
```python
class AsymmetricReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int):
        # Initialize separate buffers
        
    def add_primary(self, transition: Dict):
        # Add to primary buffer only
        
    def add_auxiliary(self, agent_id: int, transition: Dict):
        # Add to specific auxiliary buffer
        
    def sample_primary(self, batch_size: int) -> Dict:
        # Sample ONLY from primary buffer
        
    def sample_mixed(self, batch_size: int, aux_ratio: float = 0.3) -> Dict:
        # Sample from both (for policy only)
```

3. EFFICIENCY FEATURES:
   - Pre-allocated numpy arrays
   - Circular buffer implementation
   - Fast batched sampling
   - Memory-mapped storage for large buffers

4. TRANSITION FORMAT:
   - obs: [n_agents, obs_dim]
   - state: [state_dim]
   - actions: [n_agents]
   - rewards: [n_agents]
   - next_obs: [n_agents, obs_dim]
   - next_state: [state_dim]
   - done: bool
   - available_actions: [n_agents, action_dim]
```

---

## 📝 PROMPT 5: Mutual Information Estimator

```
Create the file `utils/mi_estimator.py` for efficient MI estimation in AMAL:

1. THEORETICAL FOUNDATION:
   - Estimate I(θ; O) between world model params and observations
   - Use importance sampling approximation
   - Batched computation for efficiency

2. IMPLEMENTATION:
```python
class MutualInformationEstimator:
    def __init__(self, n_samples: int = 50, n_policies: int = 10):
        # Initialize estimator parameters
        
    def estimate_mi(
        self,
        world_model: nn.Module,
        policy: nn.Module,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> float:
        # Compute MI estimate
        
    def _compute_log_prob(self, ...):
        # Helper for probability computation
        
    def _compute_mixture_log_prob(self, ...):
        # Helper for mixture approximation
```

3. EFFICIENCY TRICKS:
   - Reuse policy samples across batches
   - Stable log-sum-exp computation
   - Gradient-free estimation
   - Optional caching of policy perturbations

4. MATHEMATICAL DETAILS:
   - MI = E[log p(o|θ,π) - log Σ p(o|θ,π_k)/K]
   - Use K perturbed policies for mixture
   - M observation samples for expectation
```

---

## 📝 PROMPT 6: Main Training Script

```
Create the file `run_experiment.py` as the main entry point:

1. SCRIPT STRUCTURE:
```python
import argparse
from pathlib import Path
import yaml
import wandb

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['amal', 'mappo'])
    parser.add_argument('--scenario', default='3m')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    
    # Load configs
    # Initialize environment
    # Create algorithm
    # Training loop
    # Evaluation
    # Save results

def train_episode(env, algorithm, replay_buffer):
    # Single episode training
    
def evaluate(env, algorithm, n_episodes=32):
    # Evaluation protocol
```

2. TRAINING LOOP:
   - Collect episode with exploration
   - Store in appropriate buffer
   - Update after each episode
   - Log metrics to wandb
   - Save checkpoints periodically

3. EVALUATION:
   - Test every 100 episodes
   - Run 32 test episodes
   - Report win rate and return
   - No exploration during eval

4. METRICS TRACKING:
   - Win rate (primary metric)
   - Episode return
   - Sample efficiency
   - Training time
   - Memory usage
```

---

## 📝 PROMPT 7: Configuration Files

```
Create YAML configuration files in `configs/` directory:

1. configs/amal_config.yaml:
```yaml
algorithm:
  name: "AMAL"
  
  # World model
  world_model:
    hidden_dims: [256, 256, 128]
    learning_rate: 3e-4
    
  # Policy network
  policy:
    hidden_dims: [256, 256]
    learning_rate: 3e-4
    
  # AMAL specific
  lambda_info: 0.3
  num_auxiliary_agents: 16
  auxiliary_buffer_size: 10000
  
  # CEM evolution
  cem_population_size: 100
  cem_elite_fraction: 0.2
  evolution_frequency: 10
  
  # Training
  batch_size: 256
  gamma: 0.99
  buffer_size: 100000
```

2. configs/mappo_config.yaml:
```yaml
algorithm:
  name: "MAPPO"
  
  # Networks
  actor:
    hidden_dims: [256, 256]
    learning_rate: 5e-4
    
  critic:
    hidden_dims: [256, 256]
    learning_rate: 5e-4
    
  # PPO parameters
  clip_param: 0.2
  value_clip: 0.2
  entropy_coef: 0.01
  max_grad_norm: 10.0
  
  # Training
  n_epochs: 4
  n_minibatch: 4
  gamma: 0.99
  gae_lambda: 0.95
  buffer_size: 32  # episodes
```

3. configs/smac_scenarios.yaml:
```yaml
scenarios:
  easy:
    - "3m"
    - "2s3z"
    
  medium:
    - "8m"
    - "5m_vs_6m"
    - "2s_vs_1sc"
    
  hard:
    - "MMM"
    - "3s5z"
    
  super_hard:
    - "corridor"
    - "6h_vs_8z"
    
test_episodes: 32
parallel_envs: 8
```
```

---

## 📝 PROMPT 8: Results Visualization

```
Create the file `plot_results.py` for experiment visualization:

1. PLOTTING FUNCTIONS:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_win_rate_comparison(results_dir: Path):
    # Compare AMAL vs MAPPO win rates
    
def plot_sample_efficiency(results_dir: Path):
    # Episodes to reach 90% performance
    
def plot_learning_curves(results_dir: Path):
    # Training curves for both algorithms
    
def create_summary_table(results_dir: Path):
    # LaTeX table for paper
```

2. VISUALIZATIONS:
   - Win rate over episodes
   - Sample efficiency comparison
   - Final performance table
   - Computational cost analysis
   - Per-scenario breakdown

3. STATISTICAL TESTS:
   - Confidence intervals (bootstrap)
   - Significance testing (t-test)
   - Effect size (Cohen's d)
```

---

## 💡 Testing Strategy

After implementing each component:

1. **Unit test each module**:
```bash
python -c "from core.amal_agent import AMALAgent; print('AMAL import successful')"
python -c "from core.mappo_baseline import MAPPO; print('MAPPO import successful')"
```

2. **Test on simplest scenario first**:
```bash
# Quick smoke test
python run_experiment.py --algo amal --scenario 3m --episodes 10 --seed 0
python run_experiment.py --algo mappo --scenario 3m --episodes 10 --seed 0
```

3. **Profile before scaling**:
```bash
# Check GPU memory and speed
python run_experiment.py --algo amal --scenario 3m --episodes 100 --profile
```

4. **Run comparison**:
```bash
# Full comparison on easy scenarios
python run_experiment.py --compare --scenarios 3m,2s3z --episodes 1000
```

---

## 🚨 Common Issues and Solutions

1. **SMAC installation**: Use `pip install pymarl2` or build from source
2. **GPU memory**: Reduce batch_size or hidden_dims if OOM
3. **Slow training**: Use parallel environments (8-16 instances)
4. **Divergence**: Lower learning rates, check gradient norms
5. **Poor performance**: Verify action masking is working correctly
