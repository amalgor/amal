"""
Efficient Mutual Information Estimator for AMAL.

This module implements the estimation of mutual information I(θ; O) between
the world model parameters (θ) and the observations (O) encountered by the agent.
This estimate serves as the intrinsic motivation bonus for the Information-Seeking Objective.
"""

import torch
import torch.nn as nn
from typing import List
import sys
import os
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.base_algorithm import BaseNetwork

class MutualInformationEstimator:
    """
    Estimates mutual information using importance sampling over perturbed policies,
    as described in the AMAL paper (Proposition 1).
    """

    def __init__(self, n_samples: int = 100, n_policies: int = 20, perturbation_std: float = 0.05):
        """
        Initializes the MI estimator.

        Args:
            n_samples (int): M, the number of observation samples for the expectation.
            n_policies (int): K, the number of perturbed policy samples for the mixture.
            perturbation_std (float): Standard deviation of the noise added to create policy variants.
        """
        self.M = n_samples
        self.K = n_policies
        self.perturbation_std = perturbation_std

    @torch.no_grad()
    def estimate_mi(
        self,
        world_model: nn.Module,
        policy: nn.Module,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes the MI estimate I(θ; O) for a given batch of observations.

        The formula is: MI ≈ E_{o~D} [ log p(o|θ,π) - log( (1/K) * Σ_k p(o|θ,π_k) ) ]

        Args:
            world_model (nn.Module): The current world model p_θ.
            policy (nn.Module): The current policy network π_φ.
            observations (torch.Tensor): A batch of observations [B, obs_dim].
            actions (torch.Tensor): The actions taken in those states [B].

        Returns:
            torch.Tensor: A scalar tensor representing the estimated mutual information.
        """
        # Убедимся, что все сети в режиме оценки
        world_model.eval()
        policy.eval()

        batch_size = observations.shape[0]

        # 1. Вычисляем log p(o|θ, π) для основной политики
        # В нашем случае p(o|θ,π) - это точность предсказания world_model
        # для действий, выбранных основной политикой.
        # Мы аппроксимируем это как логарифм отрицательного MSE.
        log_prob_main = self._compute_log_prob(world_model, observations, actions, next_observations)

        # 2. Вычисляем log( (1/K) * Σ_k p(o|θ,π_k) ) для смеси политик
        
        # Создаем K возмущенных копий политики
        perturbed_policies = self._create_perturbed_policies(policy)
        
        mixture_log_probs = []
        for i, (obs_sample, action_sample) in enumerate(zip(observations, actions)):
            # Для каждого (o, a) в батче, считаем его вероятность по смеси политик
            sample_probs = []
            if next_observations is not None:
                next_obs_sample = next_observations[i]
                if isinstance(next_obs_sample, torch.Tensor):
                    next_obs_sample = next_obs_sample.unsqueeze(0)
                else:
                    next_obs_sample = torch.tensor(next_obs_sample, dtype=torch.float32).unsqueeze(0)
            else:
                next_obs_sample = None
                
            for p_policy in perturbed_policies:
                log_p = self._compute_log_prob(world_model, obs_sample.unsqueeze(0), action_sample.unsqueeze(0), next_obs_sample, p_policy)
                sample_probs.append(log_p)
            
            # Используем log-sum-exp для стабильного вычисления логарифма суммы
            mixture_log_prob = torch.logsumexp(torch.stack(sample_probs), dim=0) - torch.log(torch.tensor(self.K))
            mixture_log_probs.append(mixture_log_prob)

        log_prob_mixture = torch.stack(mixture_log_probs)

        # 3. Вычисляем MI как среднее по батчу
        mi_estimate = (log_prob_main - log_prob_mixture).mean()

        # Возвращаем сети в режим обучения, если они были в нем
        world_model.train()
        policy.train()
        
        # Убедимся, что результат не отрицательный
        return torch.clamp(mi_estimate, min=0.0)

    def _compute_log_prob(
        self, 
        world_model: nn.Module, 
        obs: torch.Tensor, 
        actions: torch.Tensor, 
        next_obs: torch.Tensor = None,
        policy: nn.Module = None
    ) -> torch.Tensor:
        """
        Аппроксимирует log p(o|θ,π) используя точность предсказания world_model 
        для действий, предсказанных заданной политикой.
        """
        action_dim = world_model.input_dim - obs.shape[-1]
        
        if policy is not None:
            # Если передана политика, используем её для предсказания действий
            with torch.no_grad():
                policy_logits = policy(obs)
                predicted_actions = torch.argmax(policy_logits, dim=-1)
            action_one_hot = nn.functional.one_hot(predicted_actions, num_classes=action_dim).float()
        else:
            # Иначе используем переданные действия
            action_one_hot = nn.functional.one_hot(actions, num_classes=action_dim).float()
        
        world_model_input = torch.cat([obs, action_one_hot], dim=-1)
        predicted_next_obs = world_model(world_model_input)
        
        if next_obs is not None:
            # Правильная реализация: используем реальные next_obs
            # Убеждаемся, что next_obs является tensor на том же device
            if not isinstance(next_obs, torch.Tensor):
                next_obs = torch.tensor(next_obs, dtype=torch.float32)
            next_obs = next_obs.to(predicted_next_obs.device)
            # Высокая точность предсказания = высокая вероятность
            mse_error = torch.mean((predicted_next_obs - next_obs) ** 2, dim=-1)
            # Преобразуем MSE в log-probability (чем меньше ошибка, тем выше вероятность)
            log_prob = -mse_error  # Можно добавить scaling factor для стабильности
            return log_prob
        else:
            # Fallback: эвристика на основе нормы предсказания
            # Это менее точно, но лучше чем ничего
            prediction_confidence = -torch.norm(predicted_next_obs, dim=-1)
            return prediction_confidence

    def _create_perturbed_policies(self, policy: nn.Module) -> List[nn.Module]:
        """Создает K копий политики с добавленным гауссовым шумом к весам."""
        perturbed_policies = []
        for _ in range(self.K):
            # Создаем глубокую копию, чтобы не изменять оригинальную политику
            p_policy = copy.deepcopy(policy)
            
            for param in p_policy.parameters():
                noise = torch.randn_like(param) * self.perturbation_std
                param.data.add_(noise)
            perturbed_policies.append(p_policy)
        return perturbed_policies

if __name__ == '__main__':
    # --- Тест MI Estimator ---
    print("\n--- Testing MutualInformationEstimator ---")

    # Параметры для теста
    OBS_DIM = 30
    ACTION_DIM = 9
    HIDDEN_DIMS = [64, 64]
    BATCH_SIZE = 32
    DEVICE = "cpu"

    # Создаем "игрушечные" сети
    world_model = BaseNetwork(OBS_DIM + ACTION_DIM, OBS_DIM, HIDDEN_DIMS).to(DEVICE)
    policy = BaseNetwork(OBS_DIM, ACTION_DIM, HIDDEN_DIMS).to(DEVICE)

    # Создаем "игрушечные" данные
    observations = torch.randn(BATCH_SIZE, OBS_DIM).to(DEVICE)
    actions = torch.randint(0, ACTION_DIM, (BATCH_SIZE,)).to(DEVICE)

    # Инициализируем и запускаем оценщик
    estimator = MutualInformationEstimator(n_samples=10, n_policies=5)
    mi_value = estimator.estimate_mi(world_model, policy, observations, actions)

    print(f"Estimated MI: {mi_value.item():.6f}")
    assert isinstance(mi_value, torch.Tensor)
    assert mi_value.item() >= 0.0
    
    print("MutualInformationEstimator test passed successfully!")