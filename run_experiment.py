"""
Main entry point for running AMAL and MAPPO experiments on SMAC.
"""

import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import wandb
import sys
import os
import pandas as pd

# Добавляем корневую директорию проекта в путь, чтобы работали импорты
sys.path.append(str(Path(__file__).resolve().parent))

from environments.smac_wrapper import SMACWrapper
from core.amal_agent import AMALAgent
from core.mappo_baseline import MAPPO
# AsymmetricReplayBuffer больше не используется напрямую в этом файле для MAPPO
# from utils.replay_buffer import AsymmetricReplayBuffer 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices=['amal', 'mappo'], required=True, help="Algorithm to run.")
    parser.add_argument('--scenario', type=str, default='3m', help="SMAC scenario to run.")
    parser.add_argument('--episodes', type=int, default=2000, help="Total episodes to train.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID.")
    parser.add_argument('--log_to_wandb', action='store_true', help="Log results to Weights & Biases.")
    args = parser.parse_args()

    # --- Загрузка конфигураций ---
    config_path = Path(__file__).resolve().parent / "configs"
    if args.algo == 'amal':
        with open(config_path / "amal_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path / "mappo_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    
    # --- Настройка окружения ---
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    env = SMACWrapper(scenario_name=args.scenario, seed=args.seed)
    
    # --- Инициализация W&B (если нужно) ---
    if args.log_to_wandb:
        wandb.init(
            project="AMAL_vs_MAPPO_SMAC",
            name=f"{args.algo}_{args.scenario}_seed{args.seed}",
            config=config,
            # Добавим reinit для возможных будущих запусков в jupyter
            reinit=True 
        )
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f"{args.algo}_{args.scenario}_seed{args.seed}.csv"
    log_data_list = []


    # --- Создание алгоритма ---
    algo_class = AMALAgent if args.algo == 'amal' else MAPPO
    algorithm = algo_class(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=config['algorithm'],
        device=device
    )

    # --- Тренировочный цикл ---
    total_steps = 0

    if args.algo == 'amal':
        # --- Off-policy цикл для AMAL ---
        for episode in range(args.episodes):
            obs, state = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            
            while not done and episode_steps < env.episode_limit:
                avail_actions = env.get_available_actions()
                actions, info = algorithm.select_actions(obs, state, avail_actions)
                
                next_obs, next_state, reward, done, env_info = env.step(actions)
                
                # AMAL использует свой внутренний асимметричный буфер
                algorithm.add_experience(
                    obs, state, actions, reward, next_obs, next_state, 
                    done, avail_actions, info.get('log_probs', {})
                )

                obs, state = next_obs, next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

            # --- Обновление и логирование ---
            if len(algorithm.replay_buffer) > config['algorithm']['batch_size']:
                 losses = algorithm.update(total_steps)
            else:
                losses = {}
            
            print(f"Episode {episode}: Steps={episode_steps}, Reward={episode_reward:.2f}, Battle Won={env_info.get('battle_won', False)}")

            if args.log_to_wandb:
                log_data = {
                    "episode": episode,
                    "reward": episode_reward,
                    "steps": episode_steps,
                    "battle_won": 1.0 if env_info.get('battle_won', False) else 0.0,
                }
                log_data.update(losses)
                wandb.log(log_data)
                log_data_list.append(log_data)
            
            # --- Периодическая оценка и сохранение ---
            if episode % 100 == 0 and episode > 0:
                save_path = f"models/{args.algo}_{args.scenario}_e{episode}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                algorithm.save(save_path)

    elif args.algo == 'mappo':
        # --- On-policy цикл для MAPPO ---
        on_policy_buffer = []
        mappo_batch_size = config['algorithm']['buffer_size'] # episodes
        
        for episode in range(args.episodes):
            obs, state = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            
            # Собираем данные для одного эпизода
            episode_data = {key: [] for key in [
                "obs", "state", "actions", "rewards", "next_obs", "next_state",
                "dones", "available_actions", "log_probs"
            ]}

            while not done and episode_steps < env.episode_limit:
                avail_actions = env.get_available_actions()
                actions, info = algorithm.select_actions(obs, state, avail_actions)
                
                next_obs, next_state, reward, done, env_info = env.step(actions)

                # Сохраняем все данные шага
                episode_data["obs"].append([obs[i] for i in range(env.n_agents)])
                episode_data["state"].append(state)
                episode_data["actions"].append([actions[i] for i in range(env.n_agents)])
                episode_data["rewards"].append([reward] * env.n_agents)
                episode_data["next_obs"].append([next_obs[i] for i in range(env.n_agents)])
                episode_data["next_state"].append(next_state)
                episode_data["dones"].append(done)
                episode_data["available_actions"].append([avail_actions[i] for i in range(env.n_agents)])
                episode_data["log_probs"].append([info['log_probs'][i] for i in range(env.n_agents)])

                obs, state = next_obs, next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
            
            on_policy_buffer.append(episode_data)

            print(f"Episode {episode}: Steps={episode_steps}, Reward={episode_reward:.2f}, Battle Won={env_info.get('battle_won', False)}")
            
            # --- Обновление после сбора батча эпизодов ---
            if len(on_policy_buffer) >= mappo_batch_size:
                # --- Форматирование батча ---
                batch = {}
                max_len = max(len(ep['obs']) for ep in on_policy_buffer)
                
                for key in episode_data.keys():
                    padded_episodes = []
                    for ep_data in on_policy_buffer:
                        ep_len = len(ep_data[key])
                        # Дополняем нулями до максимальной длины
                        pad_width = [(0, max_len - ep_len)] + [(0, 0)] * (np.array(ep_data[key]).ndim - 1)
                        padded = np.pad(ep_data[key], pad_width, 'constant', constant_values=0)
                        padded_episodes.append(padded)
                    batch[key] = torch.tensor(np.array(padded_episodes), dtype=torch.float32).to(device)

                losses = algorithm.update(batch)
                on_policy_buffer.clear() # Очищаем буфер после обновления
            else:
                losses = {}

            if args.log_to_wandb:
                log_data = {
                    "episode": episode,
                    "reward": episode_reward,
                    "steps": episode_steps,
                    "battle_won": 1.0 if env_info.get('battle_won', False) else 0.0,
                }
                log_data.update(losses)
                wandb.log(log_data)
                log_data_list.append(log_data)
            
            # --- Периодическая оценка и сохранение ---
            if episode % 100 == 0 and episode > 0:
                save_path = f"models/{args.algo}_{args.scenario}_e{episode}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                algorithm.save(save_path)

    env.close()
    if args.log_to_wandb and log_data_list:
        wandb.finish()
        results_df = pd.DataFrame(log_data_list)
        results_df.to_csv(log_path, index=False)
        print(f"Results saved to {log_path}")

if __name__ == '__main__':
    main()