#!/usr/bin/env python3
"""
Скрипт для записи лучших эпизодов AMAL в replay format
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from datetime import datetime
import shutil

# Добавляем путь к PyMARL в sys.path
PYMARL_PATH = str(Path(__file__).resolve().parent / "pymarl" / "src")
if PYMARL_PATH not in sys.path:
    sys.path.append(PYMARL_PATH)

from smac.env import StarCraft2Env
from core.amal_agent import AMALAgent

class ReplayRecorder:
    """Класс для записи и организации replay файлов"""
    
    def __init__(self, scenario_name: str, model_path: str = None):
        self.scenario_name = scenario_name
        self.model_path = model_path
        self.replay_dir = "collected_replays"
        os.makedirs(self.replay_dir, exist_ok=True)
        
        # Загружаем конфигурацию
        with open('configs/amal_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
    
    def record_episode(self, episode_type: str = "demo", max_steps: int = None):
        """Записывает один эпизод с replay"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_prefix = f"amal_{self.scenario_name}_{episode_type}_{timestamp}"
        
        # Создаем SMAC environment с replay
        env = StarCraft2Env(
            map_name=self.scenario_name,
            seed=42,
            replay_dir="/tmp",  # Временная директория
            replay_prefix=replay_prefix
        )
        
        env_info = env.get_env_info()
        n_agents = env_info["n_agents"]
        obs_dim = env_info["obs_shape"]
        state_dim = env_info["state_shape"]
        action_dim = env_info["n_actions"]
        episode_limit = env_info["episode_limit"]
        
        if max_steps is None:
            max_steps = episode_limit
        
        # Создаем агента
        agent = AMALAgent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            config=self.config['algorithm'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Загружаем модель если указана
        use_trained = False
        if self.model_path and os.path.exists(self.model_path):
            try:
                agent.load(self.model_path)
                use_trained = True
                print(f"✅ Загружена модель: {self.model_path}")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
        
        # Запускаем эпизод
        env.reset()
        obs_list = env.get_obs()
        state = env.get_state()
        obs = {i: np.array(obs_list[i]) for i in range(n_agents)}
        
        done = False
        episode_reward = 0
        episode_steps = 0
        
        print(f"🎮 Запись эпизода '{episode_type}' для {self.scenario_name}...")
        
        action_history = []
        reward_history = []
        
        while not done and episode_steps < max_steps:
            # Получаем доступные действия
            avail_actions_list = env.get_avail_actions()
            avail_actions = {i: np.array(avail_actions_list[i]) for i in range(n_agents)}
            
            if use_trained:
                # Используем обученную модель
                actions, info = agent.select_actions(obs, state, avail_actions, explore=False)
            else:
                # Случайные действия
                actions = {}
                for i in range(n_agents):
                    avail_indices = np.where(avail_actions[i] == 1)[0]
                    if len(avail_indices) > 0:
                        actions[i] = np.random.choice(avail_indices)
                    else:
                        actions[i] = 0
            
            # Выполняем шаг
            action_list = [actions.get(i, 0) for i in range(n_agents)]
            reward, done, env_info = env.step(action_list)
            
            # Обновляем состояние
            next_obs_list = env.get_obs()
            next_state = env.get_state()
            obs = {i: np.array(next_obs_list[i]) for i in range(n_agents)}
            state = np.array(next_state)
            
            # Сохраняем историю
            action_history.append(actions.copy())
            reward_history.append(reward)
            
            episode_reward += reward
            episode_steps += 1
            
            if episode_steps % 20 == 0:
                print(f"  Шаг {episode_steps}: Actions={actions}, Reward={reward:.2f}")
        
        # Сохраняем replay
        try:
            env.save_replay()
            replay_saved = True
            print("✅ Replay сохранен в SMAC")
        except Exception as e:
            print(f"❌ Ошибка сохранения replay: {e}")
            replay_saved = False
        
        env.close()
        
        # Ищем созданный файл replay
        replay_file = None
        if replay_saved:
            # Ищем файл в возможных местах
            search_paths = [
                f"/home/user/StarCraftII/Replays/{replay_prefix}*.SC2Replay",
                f"/tmp/{replay_prefix}*.SC2Replay",
                f"/home/user/StarCraftII/Replays/*/{replay_prefix}*.SC2Replay"
            ]
            
            import glob
            for pattern in search_paths:
                files = glob.glob(pattern)
                if files:
                    replay_file = files[0]
                    break
        
        # Копируем replay в нашу папку
        final_replay_path = None
        if replay_file and os.path.exists(replay_file):
            final_name = f"{replay_prefix}_r{episode_reward:.2f}_s{episode_steps}.SC2Replay"
            final_replay_path = os.path.join(self.replay_dir, final_name)
            shutil.copy2(replay_file, final_replay_path)
            print(f"📁 Replay скопирован в: {final_replay_path}")
        
        # Сохраняем метаданные
        metadata = {
            'episode_type': episode_type,
            'scenario': self.scenario_name,
            'model_path': self.model_path,
            'use_trained': use_trained,
            'episode_steps': episode_steps,
            'episode_reward': episode_reward,
            'battle_won': env_info.get('battle_won', False),
            'timestamp': timestamp,
            'replay_file': final_replay_path,
            'action_history': action_history[:10],  # Первые 10 действий для анализа
            'reward_history': reward_history[:10]
        }
        
        # Сохраняем JSON с метаданными
        import json
        metadata_file = os.path.join(self.replay_dir, f"{replay_prefix}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return metadata

def record_multiple_episodes():
    """Записывает несколько эпизодов разных типов"""
    
    print("🎬 === МАССОВАЯ ЗАПИСЬ REPLAY ЭПИЗОДОВ ===")
    
    scenarios = ["2s_vs_1sc"]  # Начнем с простого
    model_paths = [
        "models/amal_2s_vs_1sc_e100.pt",
        None  # Случайная политика для сравнения
    ]
    
    recorder_results = []
    
    for scenario in scenarios:
        for i, model_path in enumerate(model_paths):
            model_type = "trained" if model_path else "random"
            
            print(f"\n📹 Записываем {scenario} с {model_type} моделью...")
            
            recorder = ReplayRecorder(scenario, model_path)
            
            # Записываем несколько эпизодов
            for episode_num in range(3):  # 3 эпизода каждого типа
                try:
                    metadata = recorder.record_episode(
                        episode_type=f"{model_type}_{episode_num}",
                        max_steps=200  # Ограничиваем для быстроты
                    )
                    recorder_results.append(metadata)
                    
                    print(f"  ✅ Эпизод {episode_num}: R={metadata['episode_reward']:.2f}, Steps={metadata['episode_steps']}")
                    
                except Exception as e:
                    print(f"  ❌ Ошибка в эпизоде {episode_num}: {e}")
    
    # Генерируем отчет
    print(f"\n📊 === ОТЧЕТ О ЗАПИСАННЫХ REPLAY ===")
    print(f"Всего записано эпизодов: {len(recorder_results)}")
    
    for result in recorder_results:
        print(f"  {result['episode_type']}: R={result['episode_reward']:.2f}, "
              f"Steps={result['episode_steps']}, Won={result['battle_won']}")
    
    print(f"\n📁 Все replay файлы сохранены в: collected_replays/")
    print(f"\n🎮 Как просмотреть:")
    print(f"  1. Скопируйте .SC2Replay файлы в StarCraft II/Replays/")
    print(f"  2. Откройте StarCraft II → Replays")
    print(f"  3. Анализируйте поведение агентов!")
    
    return recorder_results

if __name__ == "__main__":
    try:
        results = record_multiple_episodes()
        print(f"\n🎉 Запись завершена! Записано {len(results)} эпизодов.")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
