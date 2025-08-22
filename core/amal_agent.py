"""
Core implementation of the AMAL (Asymmetric Multi-Agent Learning) algorithm.
Inherits from the BaseMAAlgorithm and implements the core logic described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List
from torch.distributions import Categorical
import wandb

# Предполагается, что эти файлы будут созданы на следующих шагах
from .base_algorithm import BaseMAAlgorithm, BaseNetwork
from utils.replay_buffer import AsymmetricReplayBuffer
from utils.mi_estimator import MutualInformationEstimator

class AMALAgent(BaseMAAlgorithm):
    """
    AMAL: Asymmetric Multi-Agent Learning Agent.

    Implements the core logic of the AMAL framework:
    1. Asymmetric Update Rule: World model learns only from the primary agent.
    2. Information-Seeking Objective: Policy loss is augmented with a mutual information bonus.
    3. Auxiliary Agent Evolution: A population of auxiliary agents is evolved via CEM to
       generate diverse experiences for the primary agent's policy learning.
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
        # 1. Модель мира (World Model)
        self.world_model = BaseNetwork(
            input_dim=obs_dim + action_dim,
            output_dim=obs_dim, # Предсказывает следующее наблюдение
            hidden_dims=self.config['world_model']['hidden_dims']
        ).to(self.device)

        # 2. Политика (Actor) и Критик (Critic)
        self.actor = BaseNetwork(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=self.config['policy']['hidden_dims']
        ).to(self.device)
        
        self.critic = BaseNetwork(
            input_dim=state_dim, # Критик использует глобальное состояние
            output_dim=1,
            hidden_dims=self.config['policy']['hidden_dims']
        ).to(self.device)

        # --- Оптимизаторы ---
        self.world_model_optimizer = optim.Adam(
            self.world_model.parameters(), 
            lr=self.config['world_model']['learning_rate']
        )
        self.policy_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config['policy']['learning_rate']
        )

        # --- Ключевые компоненты AMAL ---
        self.replay_buffer = AsymmetricReplayBuffer(
            capacity=self.config['buffer_size'],
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_aux_agents=self.config['num_auxiliary_agents']
        )
        
        self.mi_estimator = MutualInformationEstimator(
            n_samples=self.config['mi_estimator_samples_M'],
            n_policies=self.config['mi_estimator_policies_K']
        )

        # --- Управление вспомогательными агентами ---
        # Версия "B": обучаем одну общую политику для всех агентов, чтобы упростить отладку.
        # Сохраняем значение num_aux_agents, но отключаем отдельные сети.
        self.num_aux_agents = 0
        self.aux_agents = []
        # --- Инициализация параметров для CEM ---
        # Мы будем поддерживать распределение (среднее и ковариацию) для весов вспомогательных агентов
        self.cem_mu = self._get_params_vec(self.actor) # Начинаем с параметров основного агента
        self.cem_sigma = torch.ones_like(self.cem_mu) * self.config.get('cem_init_sigma', 1.0)
        
        self.cem_population_size = self.config['cem_population_size']
        self.cem_elite_fraction = self.config['cem_elite_fraction']
        self.num_elites = int(self.cem_population_size * self.cem_elite_fraction)

        # Добавим логирование в wandb
        self.use_wandb = True
        try:
            import wandb
        except ImportError:
            self.use_wandb = False


        print("AMAL Agent Initialized successfully.")

    def _get_params_vec(self, net: nn.Module) -> torch.Tensor:
        """Вспомогательная функция для преобразования параметров сети в один вектор."""
        return torch.cat([p.data.view(-1) for p in net.parameters()])

    def _set_params_from_vec(self, net: nn.Module, params_vec: torch.Tensor):
        """Вспомогательная функция для загрузки параметров в сеть из вектора."""
        offset = 0
        for p in net.parameters():
            numel = p.data.numel()
            p.data.copy_(params_vec[offset:offset + numel].view_as(p.data))
            offset += numel
            
    def select_actions(
        self,
        observations: Dict[int, np.ndarray],
        global_state: np.ndarray = None,
        available_actions: Dict[int, np.ndarray] = None,
        explore: bool = True
    ) -> Tuple[Dict[int, int], Dict[str, Any]]:
        """
        Выбирает действия для основного и вспомогательных агентов.
        Основной агент (id=0) использует self.actor.
        Вспомогательные агенты (id > 0) используют свои сети.
        """
        actions = {}
        log_probs = {}
        
        # Переводим сети в нужный режим (оценка или обучение)
        if explore:
            self.actor.train()
            for aux_agent in self.aux_agents:
                aux_agent.train()
        else:
            self.actor.eval()
            for aux_agent in self.aux_agents:
                aux_agent.eval()

        with torch.no_grad():
            # --- Основной агент ---
            agent_id = 0
            obs_tensor = torch.tensor(observations[agent_id], dtype=torch.float32).to(self.device)
            logits = self.actor(obs_tensor)

            if available_actions and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32).to(self.device)
                logits[avail_actions_tensor == 0] = -1e10 # Маскируем недоступные действия

            dist = Categorical(logits=logits)
            action = dist.sample()
            
            actions[agent_id] = action.item()
            log_probs[agent_id] = dist.log_prob(action).item()

            # --- Остальные агенты используют ту же политику ---
            for agent_id in range(1, self.n_agents):
                if agent_id not in observations:
                    continue

                obs_tensor = torch.tensor(observations[agent_id], dtype=torch.float32).to(self.device)
                logits = self.actor(obs_tensor)

                if available_actions and agent_id in available_actions:
                    avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32).to(self.device)
                    logits[avail_actions_tensor == 0] = -1e10

                dist = Categorical(logits=logits)
                action = dist.sample()

                actions[agent_id] = action.item()
                log_probs[agent_id] = dist.log_prob(action).item()

        info = {'log_probs': log_probs}
        return actions, info

    def add_experience(self, obs, state, actions, reward, next_obs, next_state, done, available_actions, log_probs):
        """Готовит и добавляет опыт в соответствующий буфер."""
        obs_arr = np.array([obs[i] for i in range(self.n_agents)])
        actions_arr = np.array([actions[i] for i in range(self.n_agents)])
        next_obs_arr = np.array([next_obs[i] for i in range(self.n_agents)])
        avail_actions_arr = np.array([available_actions[i] for i in range(self.n_agents)])
        log_probs_arr = np.array([log_probs.get(i, 0.0) for i in range(self.n_agents)])

        transition = {
            'obs': obs_arr,
            'state': state,
            'actions': actions_arr,
            'rewards': np.array([reward]),
            'next_obs': next_obs_arr,
            'next_state': next_state,
            'dones': done,
            'available_actions': avail_actions_arr,
            'log_probs': log_probs_arr
        }
        
        # ИСПРАВЛЕНИЕ: Правильное разделение данных по буферам
        # Primary buffer = только данные от основного агента (id=0)
        # Auxiliary buffer = данные от вспомогательных агентов (id=1,2,...)
        
        # Все данные идут в primary buffer (это содержит данные всех агентов для policy обучения)
        self.replay_buffer.add_primary(transition)
        
        # Auxiliary buffer пока не используется, так как все агенты в SMAC управляются одной политикой
        # В будущем можно добавить отдельную логику для auxiliary данных
        # self.replay_buffer.add_auxiliary(transition)

    def update(self, training_steps: int) -> Dict[str, float]:
        """
        Выполняет полный шаг обновления для AMAL.
        """
        if len(self.replay_buffer) < self.config['batch_size']:
            return {} # Недостаточно данных для обучения

        world_model_loss = self._update_world_model()
        policy_loss, critic_loss, mi_bonus, entropy = self._update_policy()
        
        if training_steps % self.config['evolution_frequency'] == 0:
            if self.num_aux_agents > 0:
                self._evolve_auxiliary()
            
        return {
            "world_model_loss": world_model_loss,
            "policy_loss": policy_loss,
            "critic_loss": critic_loss,
            "mi_bonus": mi_bonus,
            "entropy": entropy
        }
    
    def _update_world_model(self) -> float:
        """
        Обновляет модель мира, используя ИСКЛЮЧИТЕЛЬНО данные от основного агента.
        Это ядро асимметричного информационного гейта.
        """
        batch = self.replay_buffer.sample_primary(self.config['batch_size'])
        
        obs = torch.tensor(batch['obs'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.int64).to(self.device)
        next_obs = torch.tensor(batch['next_obs'], dtype=torch.float32).to(self.device)
        
        # Отбираем данные только для основного агента (предположим, он всегда id=0)
        primary_agent_obs = obs[:, 0, :]
        primary_agent_action = actions[:, 0]
        primary_agent_next_obs = next_obs[:, 0, :]

        # One-hot кодирование действия для входа в модель мира
        action_one_hot = nn.functional.one_hot(primary_agent_action, self.action_dim).float()
        
        world_model_input = torch.cat([primary_agent_obs, action_one_hot], dim=-1)
        predicted_next_obs = self.world_model(world_model_input)
        
        loss = nn.functional.mse_loss(predicted_next_obs, primary_agent_next_obs)
        
        self.world_model_optimizer.zero_grad()
        loss.backward()
        self.world_model_optimizer.step()
        
        return loss.item()

    def _update_policy(self) -> Tuple[float, float, float, float]:
        """
        Обновляет политику (Actor) и оценку состояния (Critic) основного агента,
        используя Advantage Actor-Critic (A2C) с MI бонусом.
        """
        # Для обновления политики используем смешанные данные, чтобы извлечь пользу из эксплорации
        batch = self.replay_buffer.sample_mixed(self.config['batch_size'], self.config.get('aux_ratio', 0.5))
        
        # Конвертируем все данные в тензоры
        obs = torch.tensor(batch['obs'], dtype=torch.float32).to(self.device)
        state = torch.tensor(batch['state'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.int64).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        next_state = torch.tensor(batch['next_state'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32).to(self.device)
        available_actions = torch.tensor(batch['available_actions'], dtype=torch.float32).to(self.device)

        # --- Отбираем данные только для основного агента (id=0) ---
        primary_obs = obs[:, 0, :]
        primary_actions = actions[:, 0]
        # Для A2C мы используем суммарную награду команды, а не индивидуальную
        team_rewards = rewards.sum(dim=1, keepdim=True)

        # --- Critic Update ---
        # Критик оценивает ценность текущего состояния
        current_v_values = self.critic(state)
        
        # И ценность следующего состояния (для вычисления TD target)
        with torch.no_grad():
            next_v_values = self.critic(next_state)
        
        # Считаем TD target: R_t + gamma * V(S_{t+1})
        target_v_values = team_rewards + self.config['gamma'] * next_v_values * (1 - dones.unsqueeze(1))
        
        # Лосс критика - это разница между его оценкой и TD target
        critic_loss = nn.functional.mse_loss(current_v_values, target_v_values.detach())
        
        # --- Actor Update ---
        # Сначала получаем логиты и распределение для действий, совершенных в прошлом
        logits = self.actor(primary_obs)
        logits[available_actions[:, 0, :] == 0] = -1e10
        dist = Categorical(logits=logits)
        
        # Считаем log_probs для тех действий, что мы реально совершили
        log_probs = dist.log_prob(primary_actions)
        
        # Энтропия - мера случайности политики. Мы поощряем ее, чтобы агент больше исследовал.
        entropy = dist.entropy().mean()

        # Вычисляем "преимущество" (Advantage): A(s,a) = Q(s,a) - V(s)
        # В нашем случае Q(s,a) аппроксимируется как target_v_values
        advantages = (target_v_values - current_v_values).detach()

        # --- Расчет MI бонуса ---
        # Получаем next_obs для primary агента
        primary_next_obs = torch.tensor(batch['next_obs'][:, 0, :], dtype=torch.float32).to(self.device)
        
        mi_bonus = self.mi_estimator.estimate_mi(
            world_model=self.world_model,
            policy=self.actor,
            observations=primary_obs,
            actions=primary_actions,
            next_observations=primary_next_obs
        )

        # --- Комбинированный лосс Политики ---
        # Мы хотим максимизировать (log_probs * advantages) и энтропию
        # и MI бонус (information gain)
        # Поэтому лосс - это отрицательное значение этих величин
        policy_loss = - (log_probs * advantages).mean()
        
        lambda_info = self.config['lambda_info']
        entropy_coef = self.config.get('entropy_coef', 0.01) # Добавляем коэффициент для энтропии

        total_loss = policy_loss - lambda_info * mi_bonus - entropy_coef * entropy + critic_loss

        # --- Шаг оптимизации ---
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        # Применяем обрезку градиентов для стабильности
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 
                                 self.config.get('max_grad_norm', 10.0))
        self.policy_optimizer.step()
        
        return policy_loss.item(), critic_loss.item(), mi_bonus.item(), entropy.item()

    def _evolve_auxiliary(self):
        """
        Выполняет шаг эволюции для популяции вспомогательных агентов с помощью CEM.
        """
        print("Evolving auxiliary agents...")
        
        # 1. Сэмплирование популяции кандидатов из текущего распределения
        population_params = torch.distributions.Normal(self.cem_mu, self.cem_sigma).sample((self.cem_population_size,))
        
        # 2. Оценка "пригодности" (fitness) каждого кандидата
        fitness_scores = []
        
        # Используем небольшой батч для оценки, чтобы не перегружать память
        eval_batch = self.replay_buffer.sample_mixed(self.config.get('cem_eval_batch_size', 128))
        eval_obs = torch.tensor(eval_batch['obs'], dtype=torch.float32).to(self.device)[:, 0, :] # Берем obs только основного агента

        for params_vec in population_params:
            # Создаем временную сеть для оценки
            temp_agent = BaseNetwork(
                input_dim=self.obs_dim, 
                output_dim=self.action_dim, 
                hidden_dims=self.config['policy']['hidden_dims']
            ).to(self.device)
            self._set_params_from_vec(temp_agent, params_vec)
            
            # Фитнес на основе "новизны" предсказаний (ошибка модели мира)
            with torch.no_grad():
                logits = temp_agent(eval_obs)
                actions = Categorical(logits=logits).sample()
                action_one_hot = F.one_hot(actions, self.action_dim).float()
                
                world_model_input = torch.cat([eval_obs, action_one_hot], dim=-1)
                predicted_next_obs = self.world_model(world_model_input)
                
                # Используем ошибку предсказания на реальных данных как метрику новизны
                real_next_obs = torch.tensor(eval_batch['next_obs'], dtype=torch.float32).to(self.device)[:, 0, :]
                novelty_error = F.mse_loss(predicted_next_obs, real_next_obs)

            # Расстояние до основного агента для регуляризации
            distance_to_main = torch.norm(params_vec - self._get_params_vec(self.actor))
            
            beta = self.config.get('cem_beta_dist', 0.1)
            fitness = novelty_error - beta * distance_to_main
            fitness_scores.append(fitness.item())
            
        # 3. Выбор "элиты" - кандидатов с наилучшим фитнесом
        elite_indices = np.argsort(fitness_scores)[-self.num_elites:]
        elite_params = population_params[elite_indices]
        
        # 4. Обновление распределения (mu и sigma) на основе "элиты"
        self.cem_mu = elite_params.mean(dim=0)
        self.cem_sigma = elite_params.std(dim=0) + 1e-6 # Добавляем эпсилон для стабильности

        # 5. Обновляем наши рабочие вспомогательные агенты, выбирая лучших из элиты
        # Мы можем просто взять N лучших из элиты или сэмплировать из нового распределения
        for i in range(min(self.num_aux_agents, self.num_elites)):
            self._set_params_from_vec(self.aux_agents[i], elite_params[-(i+1)])
            
        if self.use_wandb:
            try:
                wandb.log({
                    "cem_best_fitness": max(fitness_scores),
                    "cem_avg_fitness": np.mean(fitness_scores),
                    "cem_mu_norm": self.cem_mu.norm().item(),
                    "cem_sigma_norm": self.cem_sigma.norm().item()
                })
            except Exception:
                # Wandb not initialized, skip logging
                pass
        print(f"Auxiliary agents evolved. New best fitness: {max(fitness_scores):.4f}")
        
    def save(self, path: str):
        # ... (сохранение state_dict для всех сетей и оптимизаторов)
        pass
    
    def load(self, path: str):
        # ... (загрузка state_dict)
        pass

    @property
    def trainable_parameters(self) -> int:
        # ... (подсчет параметров)
        return 0
    
    @property
    def memory_usage(self) -> float:
        # ... (подсчет использования памяти)
        return 0.0

    def save(self, path: str):
        """Сохраняет состояние всех сетей и оптимизаторов."""
        print(f"Saving AMAL model to {path}...")
        torch.save({
            'world_model_state_dict': self.world_model.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'world_model_optimizer_state_dict': self.world_model_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'cem_mu': self.cem_mu,
            'cem_sigma': self.cem_sigma
        }, path)
        print("Save complete.")

    def load(self, path: str):
        """Загружает состояние всех сетей и оптимизаторов."""
        print(f"Loading AMAL model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint['world_model_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.cem_mu = checkpoint['cem_mu'].to(self.device)
        self.cem_sigma = checkpoint['cem_sigma'].to(self.device)
        print("Load complete.")

    @property
    def trainable_parameters(self) -> int:
        """Возвращает общее количество обучаемых параметров."""
        main_params = sum(p.numel() for p in self.world_model.parameters() if p.requires_grad) + \
                      sum(p.numel() for p in self.actor.parameters() if p.requires_grad) + \
                      sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        
        # Мы не считаем вспомогательных агентов как "обучаемые" в традиционном смысле,
        # так как они управляются через CEM, а не градиентным спуском.
        return main_params

    @property
    def memory_usage(self) -> float:
        """Возвращает использование GPU памяти в MB (упрощенная версия)."""
        # Точная реализация требует специальных библиотек, пока возвращаем 0.0
        if self.device.type == 'cuda':
            # return torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            pass
        return 0.0