"""
Base algorithm interface for SMAC experiments
Provides common interface for AMAL and MAPPO implementations
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Any
import numpy as np


class BaseMAAlgorithm(ABC):
    """
    Base class for multi-agent algorithms in SMAC
    
    This interface ensures both AMAL and MAPPO can be evaluated fairly
    with the same evaluation protocol and metrics.
    """
    
    @abstractmethod
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        config: dict,
        device: str = "cuda"
    ):
        """
        Initialize algorithm with environment specifications
        
        Args:
            n_agents: Number of agents
            obs_dim: Observation dimension per agent
            state_dim: Global state dimension
            action_dim: Action dimension per agent
            config: Algorithm-specific configuration
            device: Device for computation
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device)
        self.training = True
        
    @abstractmethod
    def select_actions(
        self,
        observations: Dict[int, np.ndarray],
        global_state: Optional[np.ndarray] = None,
        available_actions: Optional[Dict[int, np.ndarray]] = None,
        explore: bool = True
    ) -> Tuple[Dict[int, int], Dict[str, Any]]:
        """
        Select actions for all agents given observations
        
        Args:
            observations: Dict mapping agent_id to observation array
            global_state: Optional global state (for centralized training)
            available_actions: Dict mapping agent_id to available actions mask
            explore: Whether to explore (training) or exploit (evaluation)
            
        Returns:
            actions: Dict mapping agent_id to selected action
            info: Additional information (e.g., log_probs for training)
        """
        pass
    
    @abstractmethod
    def update(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Update algorithm parameters given a batch of experiences
        
        Args:
            batch: Dictionary containing:
                - obs: Agent observations [batch, n_agents, obs_dim]
                - state: Global state [batch, state_dim]
                - actions: Selected actions [batch, n_agents]
                - rewards: Rewards [batch, n_agents]
                - next_obs: Next observations [batch, n_agents, obs_dim]
                - next_state: Next global state [batch, state_dim]
                - dones: Episode termination flags [batch]
                - available_actions: Action masks [batch, n_agents, action_dim]
                
        Returns:
            losses: Dictionary of loss values for logging
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        Load model checkpoint
        
        Args:
            path: Path to load checkpoint from
        """
        pass
    
    def train_mode(self):
        """Set algorithm to training mode"""
        self.training = True
        
    def eval_mode(self):
        """Set algorithm to evaluation mode"""
        self.training = False
    
    def to(self, device: str):
        """Move algorithm to specified device"""
        self.device = torch.device(device)
        return self
    
    @property
    @abstractmethod
    def trainable_parameters(self) -> int:
        """Return number of trainable parameters"""
        pass
    
    @property
    @abstractmethod
    def memory_usage(self) -> float:
        """Return current GPU memory usage in MB"""
        pass


class BaseNetwork(nn.Module):
    """Base network class with common functionality"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], 
                 activation: str = "relu", layer_norm: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using orthogonal initialization"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""
        return self.network(x)
