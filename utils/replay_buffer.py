"""
Asymmetric Replay Buffer for AMAL.
... (docstring без изменений) ...
"""

import numpy as np
from typing import Dict, Tuple

class AsymmetricReplayBuffer:
    """
    Асимметричный буфер воспроизведения для AMAL.

    Хранит два типа данных:
    1. `primary_buffer`: данные, полученные от основного обучаемого агента 
       во взаимодействии с реальной средой. Используются для обучения модели мира.
    2. `auxiliary_buffers`: данные, сгенерированные вспомогательными агентами.
       Используются только для обучения политики основного агента.
    """
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int, action_dim: int, num_aux_agents: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_aux_agents = num_aux_agents

        # --- Основной буфер (Primary) ---
        self.p_ptr = 0
        self.p_size = 0
        self.p_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.p_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.p_actions = np.zeros((capacity, n_agents), dtype=np.int32)
        self.p_rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.p_next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.p_next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.p_dones = np.zeros(capacity, dtype=bool)
        self.p_available_actions = np.zeros((capacity, n_agents, action_dim), dtype=np.int32)
        self.p_log_probs = np.zeros((capacity, n_agents), dtype=np.float32)
        
        # --- Вспомогательный буфер (Auxiliary) ---
        self.a_ptr = 0
        self.a_size = 0
        self.a_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.a_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a_actions = np.zeros((capacity, n_agents), dtype=np.int32)
        self.a_rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.a_next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.a_next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a_dones = np.zeros(capacity, dtype=bool)
        self.a_available_actions = np.zeros((capacity, n_agents, action_dim), dtype=np.int32)
        self.a_log_probs = np.zeros((capacity, n_agents), dtype=np.float32)

    def add_primary(self, transition: Dict):
        """Добавляет переход от основного агента."""
        self._add_to_buffer('primary', transition)

    def add_auxiliary(self, transition: Dict):
        """Добавляет переход от вспомогательного агента."""
        self._add_to_buffer('auxiliary', transition)

    def _add_to_buffer(self, buffer_type: str, transition: Dict):
        if buffer_type == 'primary':
            ptr, buffer_attr_prefix = self.p_ptr, 'p_'
        else:
            ptr, buffer_attr_prefix = self.a_ptr, 'a_'

        for key, value in transition.items():
            getattr(self, buffer_attr_prefix + key)[ptr] = value

        if buffer_type == 'primary':
            self.p_ptr = (self.p_ptr + 1) % self.capacity
            self.p_size = min(self.p_size + 1, self.capacity)
        else:
            self.a_ptr = (self.a_ptr + 1) % self.capacity
            self.a_size = min(self.a_size + 1, self.capacity)
            
    def sample_primary(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Сэмплирует данные только из основного буфера."""
        if self.p_size < batch_size: return {}
        idxs = np.random.randint(0, self.p_size, size=batch_size)
        return self._get_batch('primary', idxs)

    def sample_mixed(self, batch_size: int, aux_ratio: float = 0.5) -> Dict[str, np.ndarray]:
        """Сэмплирует данные из обоих буферов в заданной пропорции."""
        p_batch_size = int(batch_size * (1 - aux_ratio))
        a_batch_size = batch_size - p_batch_size

        p_batch = {}
        if self.p_size >= p_batch_size and p_batch_size > 0:
            p_idxs = np.random.randint(0, self.p_size, size=p_batch_size)
            p_batch = self._get_batch('primary', p_idxs)

        a_batch = {}
        if self.a_size >= a_batch_size and a_batch_size > 0:
            a_idxs = np.random.randint(0, self.a_size, size=a_batch_size)
            a_batch = self._get_batch('auxiliary', a_idxs)

        if not p_batch and not a_batch: return {}
        if not p_batch: return a_batch
        if not a_batch: return p_batch

        # Объединяем батчи
        combined_batch = {key: np.concatenate([p_batch[key], a_batch[key]], axis=0) for key in p_batch}
        return combined_batch

    def _get_batch(self, buffer_type: str, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        prefix = 'p_' if buffer_type == 'primary' else 'a_'
        batch = {}
        for key in self.__dict__.keys():
            if key.startswith(prefix) and isinstance(getattr(self, key), np.ndarray):
                clean_key = key.replace(prefix, '')
                # Убедимся, что ключ без префикса также является полем данных
                if 'p_' + clean_key in self.__dict__:
                    batch[clean_key] = getattr(self, key)[idxs]
        return batch

    def __len__(self) -> int:
        return self.p_size
        
if __name__ == '__main__':
    print("\n--- Testing AsymmetricReplayBuffer ---")
    
    CAPACITY = 100
    N_AGENTS = 3
    OBS_DIM = 30
    STATE_DIM = 48
    ACTION_DIM = 9
    NUM_AUX = 16
    BATCH_SIZE = 10
    
    buffer = AsymmetricReplayBuffer(CAPACITY, N_AGENTS, OBS_DIM, STATE_DIM, ACTION_DIM, NUM_AUX)
    
    # --- Добавляем 15 случайных переходов ---
    for i in range(15):
        transition = {
            'obs': np.random.rand(N_AGENTS, OBS_DIM),
            'state': np.random.rand(STATE_DIM),
            'actions': np.random.randint(ACTION_DIM, size=N_AGENTS),
            'rewards': np.array([float(i)]),
            'next_obs': np.random.rand(N_AGENTS, OBS_DIM),
            'next_state': np.random.rand(STATE_DIM),
            'dones': (i == 14),
            'available_actions': np.ones((N_AGENTS, ACTION_DIM)),
            'log_probs': -np.random.rand(N_AGENTS)
        }
        buffer.add_primary(transition)
        if i % 2 == 0:
            buffer.add_auxiliary(transition)

    print(f"Primary buffer size: {buffer.p_size}, Auxiliary buffer size: {buffer.a_size}")
    assert buffer.p_size == 15
    assert buffer.a_size == 8
    
    primary_batch = buffer.sample_primary(BATCH_SIZE)
    mixed_batch = buffer.sample_mixed(BATCH_SIZE, aux_ratio=0.5)
    
    print(f"Sampled primary batch of size: {primary_batch['obs'].shape[0]}")
    assert primary_batch['obs'].shape == (BATCH_SIZE, N_AGENTS, OBS_DIM)
    
    print(f"Sampled mixed batch of size: {mixed_batch['obs'].shape[0]}")
    assert mixed_batch['obs'].shape[0] == BATCH_SIZE
    
    print("\nAsymmetricReplayBuffer test passed successfully!")