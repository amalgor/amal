"""
SMAC Environment Wrapper. (Production Version)
"""

import sys
from pathlib import Path
import os

# Добавляем путь к PyMARL в sys.path
# Предполагаем, что pymarl находится в корневой директории проекта
PYMARL_PATH = str(Path(__file__).resolve().parent.parent / "pymarl" / "src")
if PYMARL_PATH not in sys.path:
    sys.path.append(PYMARL_PATH)

from smac.env import StarCraft2Env
import numpy as np
from typing import Dict, Tuple, List, Any
import traceback 

class SMACWrapper:
    # ... (код класса остается без изменений) ...
    def __init__(self, scenario_name: str, seed: int = None):
        try:
            self.env = StarCraft2Env(map_name=scenario_name, seed=seed)
        except Exception as e:
            print(f"Error initializing SMAC environment for map '{scenario_name}'.")
            raise e
        self.env_info = self.env.get_env_info()
        self.n_agents = self.env_info["n_agents"]
        self.obs_dim = self.env_info["obs_shape"]
        self.state_dim = self.env_info["state_shape"]
        self.action_dim = self.env_info["n_actions"]
        self.episode_limit = self.env_info["episode_limit"]
        print(f"SMAC Wrapper initialized for map: {scenario_name}")
        print(f"Agents: {self.n_agents}, Obs Dim: {self.obs_dim}, State Dim: {self.state_dim}, Action Dim: {self.action_dim}")

    def reset(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        self.env.reset()
        obs_list = self.env.get_obs()
        state = self.env.get_state()
        return {i: np.array(obs_list[i]) for i in range(self.n_agents)}, np.array(state)

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        action_list = [actions.get(i, 0) for i in range(self.n_agents)]
        reward, done, info = self.env.step(action_list)
        next_obs_list = self.env.get_obs()
        next_state = self.env.get_state()
        return {i: np.array(next_obs_list[i]) for i in range(self.n_agents)}, np.array(next_state), reward, done, info

    def get_available_actions(self) -> Dict[int, np.ndarray]:
        avail_actions_list = self.env.get_avail_actions()
        # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ ТИПОВ ДАННЫХ ---
        return {i: np.array(avail_actions_list[i]) for i in range(self.n_agents)}

    def get_alive_agents_ids(self) -> list:
        return [agent_id for agent_id in range(self.n_agents) if self.env.get_unit_by_id(agent_id) and self.env.get_unit_by_id(agent_id).health > 0]

    def close(self):
        self.env.close()

if __name__ == '__main__':
    print("\n--- Testing SMAC Wrapper (Production Version) ---")
    
    scenarios = ["3m", "8m", "MMM"] # <-- ВОЗВРАЩАЕМ ПОЛНЫЙ СПИСОК СЦЕНАРИЕВ
    
    for scenario in scenarios:
        print(f"\n--- Testing scenario: {scenario} ---")
        env = None
        try:
            env = SMACWrapper(scenario_name=scenario, seed=0)
            obs, state = env.reset()
            print(f"Initial obs for agent 0 shape: {obs[0].shape}")
            print(f"Initial state shape: {state.shape}")
            
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 100:
                actions = {}
                avail_actions = env.get_available_actions() # Теперь возвращает dict of numpy arrays
                alive_agents = env.get_alive_agents_ids()
                
                if not alive_agents:
                    print("All agents are dead. Ending episode.")
                    _, _, reward, done, info = env.step({})
                    episode_reward += reward
                    break

                for agent_id in alive_agents:
                    # Теперь эта операция будет работать, так как avail_actions[agent_id] - это numpy array
                    avail_indices = np.where(avail_actions[agent_id] == 1)[0]
                    
                    if len(avail_indices) > 0:
                        action = np.random.choice(avail_indices)
                        actions[agent_id] = int(action)
                
                _, _, reward, done, info = env.step(actions)
                episode_reward += reward
                steps += 1
            
            print(f"Ran {steps} steps. Final reward: {episode_reward:.2f}")
            battle_won = info.get('battle_won', 'N/A')
            print(f"Final Info: battle_won={battle_won}")

        except Exception as e:
            print(f"\n--- AN ERROR OCCURRED ---")
            print(f"Could not run test for scenario '{scenario}'. Error: {e}")
            print("\n--- FULL STACK TRACE ---")
            traceback.print_exc()
            print("\n-------------------------")
        finally:
            if env:
                env.close()