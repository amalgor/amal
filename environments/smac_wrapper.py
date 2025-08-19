"""
SMAC Environment Wrapper.

Provides a simplified and consistent interface to the StarCraft II Multi-Agent Challenge (SMAC)
environment, tailored for the needs of our MARL algorithms (AMAL and MAPPO).
"""

import numpy as np
from smac.env import StarCraft2Env
from typing import Dict, Tuple, Any, List

class SMACWrapper:
    """
    Wraps the StarCraft2Env to provide a more convenient API for training MARL agents.
    
    Key features:
    - Handles multiple SMAC scenarios (maps).
    - Converts SMAC's tuple-based observations/actions into dictionaries.
    - Manages available actions masking.
    - Provides both local observations for agents and the global state for the critic.
    - Tracks and aggregates episode statistics.
    """

    def __init__(self, scenario_name: str, seed: int = None):
        """
        Initializes the StarCraft II environment.

        Args:
            scenario_name (str): The name of the SMAC map to run (e.g., "3m", "8m").
            seed (int, optional): Random seed for the environment.
        """
        try:
            self.env = StarCraft2Env(map_name=scenario_name, seed=seed)
        except Exception as e:
            print(f"Error initializing SMAC environment for map '{scenario_name}'. Is StarCraft II installed?")
            print(f"Original error: {e}")
            raise

        self.env_info = self.env.get_env_info()
        
        # --- Извлечение ключевых параметров среды ---
        self.n_agents = self.env_info["n_agents"]
        self.obs_dim = self.env_info["obs_shape"]
        self.state_dim = self.env_info["state_shape"]
        self.action_dim = self.env_info["n_actions"]
        self.episode_limit = self.env_info["episode_limit"]

        print(f"SMAC Wrapper initialized for map: {scenario_name}")
        print(f"Agents: {self.n_agents}, Obs Dim: {self.obs_dim}, State Dim: {self.state_dim}, Action Dim: {self.action_dim}")

    def reset(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Resets the environment for a new episode.

        Returns:
            Tuple[Dict[int, np.ndarray], np.ndarray]:
                - A dictionary mapping agent_id to its local observation.
                - The global state of the environment.
        """
        self.env.reset()
        obs_list = self.env.get_obs()
        state = self.env.get_state()
        
        observations = {i: obs_list[i] for i in range(self.n_agents)}
        
        return observations, state

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes a joint action for all agents in the environment.

        Args:
            actions (Dict[int, int]): A dictionary mapping agent_id to its chosen action.

        Returns:
            Tuple[Dict[int, np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
                - next_observations: Dictionary of next local observations for each agent.
                - next_state: The next global state.
                - reward: The team reward for the step.
                - done: Boolean flag indicating if the episode has terminated.
                - info: Dictionary containing additional episode information (e.g., battle_won).
        """
        # Преобразуем словарь действий в список, который ожидает среда
        action_list = [actions[i] for i in range(self.n_agents)]
        
        reward, done, info = self.env.step(action_list)
        
        next_obs_list = self.env.get_obs()
        next_state = self.env.get_state()
        
        next_observations = {i: next_obs_list[i] for i in range(self.n_agents)}
        
        return next_observations, next_state, reward, done, info

    def get_available_actions(self) -> Dict[int, np.ndarray]:
        """
        Gets the available actions mask for each agent.

        Returns:
            Dict[int, np.ndarray]: A dictionary mapping agent_id to a binary vector
                                  where 1 indicates an available action.
        """
        avail_actions_list = self.env.get_avail_actions()
        return {i: avail_actions_list[i] for i in range(self.n_agents)}

    def get_stats(self) -> Dict[str, float]:
        """
        Returns a dictionary of aggregated statistics from the environment.
        Useful for logging at the end of an episode.
        """
        return self.env.get_stats()

    def close(self):
        """Closes the environment."""
        self.env.close()

if __name__ == '__main__':
    # --- Пример использования ---
    print("\n--- Testing SMAC Wrapper ---")
    
    # Список сценариев для тестирования из вашего плана
    scenarios = ["3m", "8m", "MMM"]
    
    for scenario in scenarios:
        print(f"\n--- Testing scenario: {scenario} ---")
        try:
            env = SMACWrapper(scenario_name=scenario)
            
            obs, state = env.reset()
            
            print(f"Initial obs for agent 0 shape: {obs[0].shape}")
            print(f"Initial state shape: {state.shape}")
            
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 10:
                avail_actions = env.get_available_actions()
                
                actions = {}
                for i in range(env.n_agents):
                    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
                    # 1. Находим индексы доступных действий
                    avail_indices = np.where(avail_actions[i] == 1)[0]
                    # 2. Выбираем одно случайное действие из этого списка
                    action = np.random.choice(avail_indices)
                    actions[i] = int(action) # Явно преобразуем в int
                
                next_obs, next_state, reward, done, info = env.step(actions)
                
                episode_reward += reward
                steps += 1

            print(f"Ran 10 steps. Final reward: {episode_reward:.2f}")
            print(f"Episode info at the end: {info}")
            stats = env.get_stats()
            # Улучшаем вывод, чтобы избежать ошибок, если ключа нет
            battle_won = info.get('battle_won', 'N/A')
            print(f"Final Stats: battle_won={battle_won}")
            
            env.close()

        except Exception as e:
            print(f"Could not run test for scenario '{scenario}'. Error: {e}")
            print("Please ensure StarCraft II and the SMAC maps are installed correctly.")
            print("See instructions at: https://github.com/oxwhirl/smac")            
