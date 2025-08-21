#!/usr/bin/env python3
"""
Демонстрация записи и воспроизведения SMAC replay
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from datetime import datetime

# Добавляем путь к PyMARL в sys.path
PYMARL_PATH = str(Path(__file__).resolve().parent / "pymarl" / "src")
if PYMARL_PATH not in sys.path:
    sys.path.append(PYMARL_PATH)

from smac.env import StarCraft2Env
from core.amal_agent import AMALAgent

class SMACReplayWrapper:
    """SMAC Wrapper с поддержкой replay"""
    
    def __init__(self, scenario_name: str, seed: int = None, 
                 replay_dir: str = None, replay_prefix: str = None):
        
        # Настраиваем директорию для replay
        if replay_dir is None:
            replay_dir = os.path.join(os.getcwd(), "replays")
        
        # Создаем директорию если не существует
        os.makedirs(replay_dir, exist_ok=True)
        
        if replay_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            replay_prefix = f"{scenario_name}_{timestamp}"
        
        print(f"🎬 Настройка replay:")
        print(f"  Директория: {replay_dir}")
        print(f"  Префикс: {replay_prefix}")
        
        try:
            self.env = StarCraft2Env(
                map_name=scenario_name, 
                seed=seed,
                replay_dir=replay_dir,
                replay_prefix=replay_prefix
            )
        except Exception as e:
            print(f"Ошибка инициализации SMAC для '{scenario_name}': {e}")
            raise e
            
        self.env_info = self.env.get_env_info()
        self.n_agents = self.env_info["n_agents"]
        self.obs_dim = self.env_info["obs_shape"]
        self.state_dim = self.env_info["state_shape"]
        self.action_dim = self.env_info["n_actions"]
        self.episode_limit = self.env_info["episode_limit"]
        
        print(f"✅ SMAC Replay Wrapper инициализирован для: {scenario_name}")
        print(f"   Агенты: {self.n_agents}, Obs: {self.obs_dim}, State: {self.state_dim}, Actions: {self.action_dim}")
    
    def reset(self):
        self.env.reset()
        obs_list = self.env.get_obs()
        state = self.env.get_state()
        return {i: np.array(obs_list[i]) for i in range(self.n_agents)}, np.array(state)
    
    def step(self, actions):
        action_list = [actions.get(i, 0) for i in range(self.n_agents)]
        reward, done, info = self.env.step(action_list)
        next_obs_list = self.env.get_obs()
        next_state = self.env.get_state()
        return {i: np.array(next_obs_list[i]) for i in range(self.n_agents)}, np.array(next_state), reward, done, info
    
    def get_available_actions(self):
        avail_actions_list = self.env.get_avail_actions()
        return {i: np.array(avail_actions_list[i]) for i in range(self.n_agents)}
    
    def save_replay(self):
        """Сохраняет replay текущего эпизода"""
        try:
            self.env.save_replay()
            print("✅ Replay сохранен!")
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения replay: {e}")
            return False
    
    def close(self):
        self.env.close()

def demo_replay_episode():
    """Демонстрация одного эпизода с записью replay"""
    
    print("🎮 === ДЕМОНСТРАЦИЯ REPLAY В SMAC ===")
    
    # Настройки
    scenario = "2s_vs_1sc"
    replay_dir = "replays"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_prefix = f"amal_demo_{timestamp}"
    
    # Создаем environment с replay
    env = SMACReplayWrapper(
        scenario_name=scenario,
        seed=42,
        replay_dir=replay_dir,
        replay_prefix=replay_prefix
    )
    
    # Загружаем обученную AMAL модель
    with open('configs/amal_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    agent = AMALAgent(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=config['algorithm'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Пробуем загрузить обученную модель
    try:
        agent.load('models/amal_2s_vs_1sc_e100.pt')
        print("✅ Загружена обученная AMAL модель")
        use_trained = True
    except:
        print("⚠️ Используем случайную политику")
        use_trained = False
    
    # Запускаем эпизод
    obs, state = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    
    print(f"\n🎯 Начинаем эпизод...")
    print(f"Цель: запись действий агентов для последующего анализа")
    
    while not done and episode_steps < env.episode_limit:
        avail_actions = env.get_available_actions()
        
        if use_trained:
            # Используем обученную модель
            actions, info = agent.select_actions(obs, state, avail_actions, explore=False)
        else:
            # Случайные действия из доступных
            actions = {}
            for i in range(env.n_agents):
                avail_indices = np.where(avail_actions[i] == 1)[0]
                if len(avail_indices) > 0:
                    actions[i] = np.random.choice(avail_indices)
                else:
                    actions[i] = 0
        
        # Выполняем шаг
        next_obs, next_state, reward, done, env_info = env.step(actions)
        
        episode_reward += reward
        episode_steps += 1
        
        # Логируем каждые 10 шагов
        if episode_steps % 10 == 0:
            print(f"  Шаг {episode_steps}: Actions={actions}, Reward={reward:.2f}")
        
        obs, state = next_obs, next_state
    
    # Сохраняем replay
    replay_saved = env.save_replay()
    
    # Результаты
    print(f"\n📊 Результаты эпизода:")
    print(f"  Шагов: {episode_steps}")
    print(f"  Общий reward: {episode_reward:.2f}")
    print(f"  Победа: {env_info.get('battle_won', False)}")
    print(f"  Replay сохранен: {replay_saved}")
    
    if replay_saved:
        replay_path = os.path.join(replay_dir, f"{replay_prefix}.SC2Replay")
        print(f"  📁 Файл replay: {replay_path}")
        print(f"\n🎬 Как просмотреть replay:")
        print(f"  1. Откройте StarCraft II")
        print(f"  2. Перейдите в раздел 'Replays'")
        print(f"  3. Найдите файл: {replay_prefix}.SC2Replay")
        print(f"  4. Или скопируйте в ~/Documents/StarCraft II/Accounts/*/Replays/")
    
    env.close()
    return replay_saved, episode_reward, episode_steps

if __name__ == "__main__":
    # Запускаем демонстрацию
    try:
        replay_saved, reward, steps = demo_replay_episode()
        
        print(f"\n🎉 Демонстрация завершена!")
        print(f"Replay записан: {replay_saved}")
        if replay_saved:
            print(f"\n💡 Теперь вы можете:")
            print(f"  - Просмотреть replay в StarCraft II")
            print(f"  - Анализировать поведение агентов")
            print(f"  - Понять, почему AMAL принимает определенные решения")
        
    except Exception as e:
        print(f"❌ Ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()
