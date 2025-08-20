"""
Asymmetric Replay Buffer for AMAL.
... (docstring без изменений) ...
"""

import numpy as np
from typing import Dict, Tuple

class AsymmetricReplayBuffer:
    # ... (код класса __init__ без изменений) ...
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32) # Награда одна на команду
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.available_actions = np.zeros((capacity, n_agents, action_dim), dtype=np.int32)
        self.log_probs = np.zeros((capacity, n_agents), dtype=np.float32)

    def add(
        self,
        obs: Dict[int, np.ndarray],
        state: np.ndarray,
        actions: Dict[int, int],
        reward: float,  # <--- ИСПРАВЛЕНИЕ: теперь это float, а не dict
        next_obs: Dict[int, np.ndarray],
        next_state: np.ndarray,
        done: bool,
        available_actions: Dict[int, np.ndarray],
        log_probs: Dict[int, float]
    ):
        """
        Добавляет один шаг (transition) в основной буфер.
        """
        obs_arr = np.array([obs[i] for i in range(self.n_agents)])
        actions_arr = np.array([actions[i] for i in range(self.n_agents)])
        next_obs_arr = np.array([next_obs[i] for i in range(self.n_agents)])
        avail_actions_arr = np.array([available_actions[i] for i in range(self.n_agents)])
        log_probs_arr = np.array([log_probs.get(i, 0.0) for i in range(self.n_agents)])
        
        self.obs[self.ptr] = obs_arr
        self.state[self.ptr] = state
        self.actions[self.ptr] = actions_arr
        self.rewards[self.ptr] = reward # <--- ИСПРАВЛЕНИЕ: сохраняем скалярную награду
        self.next_obs[self.ptr] = next_obs_arr
        self.next_state[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.available_actions[self.ptr] = avail_actions_arr
        self.log_probs[self.ptr] = log_probs_arr
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_primary(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return dict(
            obs=self.obs[idxs],
            state=self.state[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs], # Теперь это [batch_size, 1]
            next_obs=self.next_obs[idxs],
            next_state=self.next_state[idxs],
            dones=self.dones[idxs],
            available_actions=self.available_actions[idxs],
            log_probs=self.log_probs[idxs]
        )

    # ... (остальная часть класса без изменений) ...
    def add_auxiliary(self, transition: Dict):
        pass
    def sample_mixed(self, batch_size: int) -> Dict:
        return self.sample_primary(batch_size)
    def __len__(self) -> int:
        return self.size


if __name__ == '__main__':
    print("\n--- Testing AsymmetricReplayBuffer ---")
    
    CAPACITY = 100
    N_AGENTS = 3
    OBS_DIM = 30
    STATE_DIM = 48
    ACTION_DIM = 9
    BATCH_SIZE = 10
    
    buffer = AsymmetricReplayBuffer(CAPACITY, N_AGENTS, OBS_DIM, STATE_DIM, ACTION_DIM)
    
    # --- Добавляем 15 случайных переходов ---
    for i in range(15):
        obs = {i: np.random.rand(OBS_DIM) for i in range(N_AGENTS)}
        state = np.random.rand(STATE_DIM)
        actions = {i: np.random.randint(ACTION_DIM) for i in range(N_AGENTS)}
        reward = float(i) # <--- ИСПРАВЛЕНИЕ: награда теперь просто float
        next_obs = {i: np.random.rand(OBS_DIM) for i in range(N_AGENTS)}
        next_state = np.random.rand(STATE_DIM)
        done = (i == 14)
        available_actions = {i: np.ones(ACTION_DIM) for i in range(N_AGENTS)}
        log_probs = {i: -np.random.rand() for i in range(N_AGENTS)}
        
        buffer.add(obs, state, actions, reward, next_obs, next_state, done, available_actions, log_probs)

    print(f"Buffer size after adding 15 transitions: {len(buffer)}")
    assert len(buffer) == 15
    
    primary_batch = buffer.sample_primary(BATCH_SIZE)
    
    print(f"Sampled a batch of size: {primary_batch['obs'].shape[0]}")
    assert primary_batch['obs'].shape == (BATCH_SIZE, N_AGENTS, OBS_DIM)
    assert primary_batch['state'].shape == (BATCH_SIZE, STATE_DIM)
    assert primary_batch['rewards'].shape == (BATCH_SIZE, 1) # Проверяем новый shape для наград
    
    print("Batch keys:", primary_batch.keys())
    print(f"Shape of 'rewards' in batch: {primary_batch['rewards'].shape}")
    
    for i in range(CAPACITY):
        buffer.add(obs, state, actions, reward, next_obs, next_state, done, available_actions, log_probs)
        
    print(f"Buffer size after overflow: {len(buffer)}")
    assert len(buffer) == CAPACITY
    
    print("\nAsymmetricReplayBuffer test passed successfully!")