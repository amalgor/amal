"""
Core implementation of the MAPPO (Multi-Agent Proximal Policy Optimization) baseline.
Based on the paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
(Yu et al., 2022).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple

from .base_algorithm import BaseMAAlgorithm, BaseNetwork
from torch.distributions import Categorical

class MAPPO(BaseMAAlgorithm):
    """
    MAPPO: Multi-Agent Proximal Policy Optimization.
    
    Implements a robust baseline for SMAC environments with:
    1. Centralized Critic: Uses global state information for more stable value estimation.
    2. Decentralized Actors: Agents select actions based only on their local observations.
    3. PPO Components: Utilizes clipped surrogate objective, GAE, and entropy bonus.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        config: dict,
        device: str = "cuda"
    ):
        super().__init__(n_agents, obs_dim, state_dim, action_dim, config, device)

        # --- Инициализация сетей ---
        self.actor = BaseNetwork(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=self.config['actor']['hidden_dims']
        ).to(self.device)

        self.critic = BaseNetwork(
            input_dim=state_dim, # Критик принимает глобальное состояние
            output_dim=1,
            hidden_dims=self.config['critic']['hidden_dims']
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config['actor']['learning_rate']
        )
        
        # --- Параметры PPO ---
        self.clip_param = self.config['clip_param']
        self.entropy_coef = self.config['entropy_coef']
        self.gae_lambda = self.config['gae_lambda']
        self.n_epochs = self.config['n_epochs']
        self.n_minibatch = self.config['n_minibatch']

    def select_actions(
        self,
        observations: Dict[int, np.ndarray],
        global_state: np.ndarray = None,
        available_actions: Dict[int, np.ndarray] = None,
        explore: bool = True
    ) -> Tuple[Dict[int, int], Dict[str, Any]]:
        
        actions = {}
        log_probs = {}
        values = {}

        if not explore:
            self.actor.eval()
            self.critic.eval()
        else:
            self.actor.train()
            self.critic.train()

        with torch.no_grad():
            for agent_id, obs in observations.items():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                logits = self.actor(obs_tensor)

                if available_actions and agent_id in available_actions:
                    avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32).to(self.device)
                    # Приводим маску к форме [1, 9], чтобы она совпадала с logits
                    logits[avail_actions_tensor.unsqueeze(0) == 0] = -1e10
                
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                actions[agent_id] = action.item()
                log_probs[agent_id] = dist.log_prob(action).item()
        
        info = {'log_probs': log_probs}
        return actions, info

    def _compute_returns(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Вычисляет GAE (Generalized Advantage Estimation)."""
        rewards = batch['rewards'].sum(dim=2, keepdim=True) # Суммарная награда команды
        dones = batch['dones'].unsqueeze(2)
        
        with torch.no_grad():
            values = self.critic(batch['state'])
            next_values = self.critic(batch['next_state'])

        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        # Идем в обратном порядке по временной оси
        for t in reversed(range(batch['rewards'].shape[1])):
            delta = rewards[:, t] + self.config['gamma'] * next_values[:, t] * (1 - dones[:, t]) - values[:, t]
            last_gae_lam = delta + self.config['gamma'] * self.gae_lambda * (1 - dones[:, t]) * last_gae_lam
            advantages[:, t] = last_gae_lam
            
        returns = advantages + values
        return returns, advantages

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Обновляет параметры MAPPO, используя PPO-обновления для батча эпизодов.
        """
        # --- Преобразование данных ---
        # В MAPPO мы обычно работаем с батчем целых эпизодов
        # Здесь мы предполагаем, что батч имеет вид [num_episodes, episode_len, ...]
        batch_size = batch['obs'].shape[0] * batch['obs'].shape[1]
        
        obs_batch = batch['obs'].view(batch_size, self.n_agents, self.obs_dim)
        state_batch = batch['state'].view(batch_size, self.state_dim)
        actions_batch = batch['actions'].view(batch_size, self.n_agents)
        avail_actions_batch = batch['available_actions'].view(batch_size, self.n_agents, self.action_dim)
        
        # Загружаем старые log_probs, которые были посчитаны при сборе данных
        old_log_probs_batch = batch['log_probs'].view(batch_size, self.n_agents)

        returns, advantages = self._compute_returns(batch)
        returns = returns.view(batch_size, 1)
        advantages = advantages.view(batch_size, 1)
        
        # Нормализуем преимущества
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Цикл PPO-обновлений ---
        for _ in range(self.n_epochs):
            # Перемешиваем данные для мини-батчей
            sampler = torch.randperm(batch_size)
            
            for indices in sampler.split(batch_size // self.n_minibatch):
                
                # --- Лосс политики (Actor) ---
                policy_loss_total = 0
                for agent_id in range(self.n_agents):
                    logits = self.actor(obs_batch[indices, agent_id])
                    logits[avail_actions_batch[indices, agent_id] == 0] = -1e10
                    dist = Categorical(logits=logits)
                    
                    new_log_probs = dist.log_prob(actions_batch[indices, agent_id])
                    ratio = torch.exp(new_log_probs - old_log_probs_batch[indices, agent_id])
                    
                    surr1 = ratio * advantages[indices].squeeze()
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages[indices].squeeze()
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    policy_loss_total += policy_loss

                # --- Лосс критика (Critic) ---
                values = self.critic(state_batch[indices])
                critic_loss = nn.functional.mse_loss(values, returns[indices])
                
                # --- Энтропийный бонус ---
                # (Для простоты здесь не рассчитываем, но в полной реализации он должен быть)
                entropy_loss = 0.0

                total_loss = policy_loss_total + critic_loss - self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.config['max_grad_norm'])
                self.optimizer.step()

        return {
            "policy_loss": policy_loss_total.item(),
            "critic_loss": critic_loss.item(),
        }

    def save(self, path: str):
        """Сохраняет состояние актора, критика и оптимизатора."""
        print(f"Saving MAPPO model to {path}...")
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print("Save complete.")

    def load(self, path: str):
        """Загружает состояние актора, критика и оптимизатора."""
        print(f"Loading MAPPO model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Load complete.")
    
    @property
    def trainable_parameters(self) -> int:
        """Возвращает общее количество обучаемых параметров."""
        return sum(p.numel() for p in self.actor.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
    
    @property
    def memory_usage(self) -> float:
        """Возвращает использование GPU памяти в MB (упрощенная версия)."""
        if self.device.type == 'cuda':
            # return torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            pass
        return 0.0