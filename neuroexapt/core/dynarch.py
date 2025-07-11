"""
DynArch: Dynamic Architecture Adjustment Framework.

Advanced RL-guided evolution for neural nets with multi-objective optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict, deque
import random
from dataclasses import dataclass
from .structural_evolution import StructuralEvolution, EvolutionStep
from .operators import PruneByEntropy, ExpandWithMI, MutateDiscrete, CompoundOperator, StructuralOperator
from .information_theory import InformationBottleneck

@dataclass
class Experience:
    """Experience tuple for replay buffer."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]

class ExperienceBuffer:
    """Experience replay buffer for better RL training."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class AttentionPolicyNetwork(nn.Module):
    """Enhanced policy network with attention mechanism."""
    def __init__(self, input_dim: int, num_actions: int = 6, hidden_dim: int = 256, device: Optional[torch.device] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Self-attention for feature refinement
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights and move to device
        self.apply(self._init_weights)
        self.to(self.device)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure state is on correct device
        state = state.to(self.device)
        
        # Handle batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Feature extraction
        features = self.feature_extractor(state)
        
        # Self-attention (treating each feature as a sequence element)
        features_seq = features.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.attention(features_seq, features_seq, features_seq)
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Policy and value outputs
        policy_logits = self.policy_head(attended_features)
        value = self.value_head(attended_features)
        
        # Apply softmax to policy
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        return policy_probs, value

class MultiObjectiveReward:
    """Multi-objective reward computation with Pareto efficiency."""
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights if weights is not None else {
            'accuracy': 0.4,
            'efficiency': 0.3,
            'information': 0.2,
            'stability': 0.1
        }
        self.pareto_front = []
        
    def compute_reward(self, prev_metrics: Dict, new_metrics: Dict, action: int) -> float:
        """Compute multi-objective reward with Pareto considerations."""
        objectives = self._compute_objectives(prev_metrics, new_metrics, action)
        
        # Weighted sum for primary reward
        primary_reward = sum(self.weights[key] * value for key, value in objectives.items())
        
        # Pareto bonus: reward for non-dominated solutions
        pareto_bonus = self._compute_pareto_bonus(objectives)
        
        # Exploration bonus based on action diversity
        exploration_bonus = self._compute_exploration_bonus(action)
        
        total_reward = primary_reward + 0.1 * pareto_bonus + 0.05 * exploration_bonus
        
        # Update Pareto front
        self._update_pareto_front(objectives)
        
        return total_reward
    
    def _compute_objectives(self, prev: Dict, new: Dict, action: int) -> Dict[str, float]:
        """Compute individual objectives."""
        # Accuracy objective
        acc_delta = new.get('accuracy', 0) - prev.get('accuracy', 0)
        accuracy_obj = np.tanh(acc_delta * 10)  # Normalized to [-1, 1]
        
        # Efficiency objective (parameter reduction)
        prev_params = prev.get('params', 1)
        new_params = new.get('params', 1)
        param_ratio = (prev_params - new_params) / max(prev_params, 1)
        efficiency_obj = np.tanh(param_ratio * 5)
        
        # Information objective (mutual information preservation)
        mi_delta = new.get('mutual_info', 0) - prev.get('mutual_info', 0)
        information_obj = np.tanh(mi_delta * 2)
        
        # Stability objective (penalize drastic changes)
        stability_penalty = abs(param_ratio) if abs(param_ratio) > 0.1 else 0
        stability_obj = 1.0 - stability_penalty
        
        return {
            'accuracy': accuracy_obj,
            'efficiency': efficiency_obj,
            'information': information_obj,
            'stability': stability_obj
        }
    
    def _compute_pareto_bonus(self, objectives: Dict[str, float]) -> float:
        """Compute bonus for Pareto-optimal solutions."""
        if not self.pareto_front:
            return 1.0  # First solution gets bonus
        
        obj_vector = list(objectives.values())
        
        # Check if current solution dominates any in Pareto front
        dominates_any = any(
            all(obj_vector[i] >= front_sol[i] for i in range(len(obj_vector))) and
            any(obj_vector[i] > front_sol[i] for i in range(len(obj_vector)))
            for front_sol in self.pareto_front
        )
        
        # Check if current solution is dominated
        is_dominated = any(
            all(front_sol[i] >= obj_vector[i] for i in range(len(obj_vector))) and
            any(front_sol[i] > obj_vector[i] for i in range(len(obj_vector)))
            for front_sol in self.pareto_front
        )
        
        if dominates_any:
            return 1.0  # High bonus for dominating solutions
        elif not is_dominated:
            return 0.5  # Medium bonus for non-dominated solutions
        else:
            return 0.0  # No bonus for dominated solutions
    
    def _compute_exploration_bonus(self, action: int) -> float:
        """Encourage exploration of diverse actions."""
        # Simple exploration bonus (can be enhanced with action history)
        return 0.1 if random.random() < 0.1 else 0.0
    
    def _update_pareto_front(self, objectives: Dict[str, float]):
        """Update Pareto front with new solution."""
        obj_vector = list(objectives.values())
        
        # Remove dominated solutions from front
        self.pareto_front = [
            front_sol for front_sol in self.pareto_front
            if not (
                all(obj_vector[i] >= front_sol[i] for i in range(len(obj_vector))) and
                any(obj_vector[i] > front_sol[i] for i in range(len(obj_vector)))
            )
        ]
        
        # Add current solution if not dominated
        is_dominated = any(
            all(front_sol[i] >= obj_vector[i] for i in range(len(obj_vector))) and
            any(front_sol[i] > obj_vector[i] for i in range(len(obj_vector)))
            for front_sol in self.pareto_front
        )
        
        if not is_dominated:
            self.pareto_front.append(obj_vector)
        
        # Limit Pareto front size
        if len(self.pareto_front) > 50:
            self.pareto_front = self.pareto_front[-50:]

class DynamicArchitecture:
    """Enhanced dynamic architecture with better RL and robust device management."""
    def __init__(
        self, 
        base_model: nn.Module, 
        evolution: StructuralEvolution, 
        ib: InformationBottleneck,
        device: Optional[torch.device] = None,
        policy_lr: float = 1e-4,
        value_lr: float = 1e-3,
        buffer_size: int = 10000,
        batch_size: int = 32,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 1000,
        gamma: float = 0.99,
        clip_ratio: float = 0.2
    ):
        self.base_model = base_model
        self.evolution = evolution
        self.ib = ib
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
        # Device management
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(self.device)
        
        # Enhanced policy network with device specification
        self.policy_net = AttentionPolicyNetwork(input_dim=8, num_actions=6, device=self.device)
        self.target_net = AttentionPolicyNetwork(input_dim=8, num_actions=6, device=self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Separate optimizers for policy and value
        self.policy_optimizer = torch.optim.AdamW(
            [p for n, p in self.policy_net.named_parameters() if 'policy' in n], 
            lr=policy_lr, weight_decay=1e-4
        )
        self.value_optimizer = torch.optim.AdamW(
            [p for n, p in self.policy_net.named_parameters() if 'value' in n], 
            lr=value_lr, weight_decay=1e-4
        )
        
        # Experience replay
        self.experience_buffer = ExperienceBuffer(buffer_size)
        self.batch_size = batch_size
        
        # Exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Multi-objective reward
        self.reward_computer = MultiObjectiveReward()
        
        # History tracking
        self.history: List[EvolutionStep] = []
        self.action_counts = {i: 0 for i in range(6)}
        
        # Operators (expanded) with device-aware initialization
        self.operators = CompoundOperator([
            PruneByEntropy(), 
            ExpandWithMI(), 
            MutateDiscrete(), 
            DeviceAwareBranchOperator(self.device),
            DeviceAwareAttentionOperator(self.device),
            DeviceAwareHybridOperator(self.device)
        ])
    
    def get_state(self, metrics: Dict[str, float]) -> torch.Tensor:
        """Enhanced state representation with device consistency."""
        state = [
            metrics.get('entropy', 0.5),
            metrics.get('mutual_info', 0.0),
            metrics.get('accuracy', 0.0),
            metrics.get('params', 0) / 1e6,  # Normalized
            metrics.get('redundancy', 0.0),
            metrics.get('complexity', 0.0),
            metrics.get('loss', 1.0),
            len(self.history) / 100.0  # Evolution step count (normalized)
        ]
        return torch.tensor(state, dtype=torch.float32, device=self.device)
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Enhanced action selection with epsilon-greedy exploration."""
        # Ensure state is on correct device
        state = state.to(self.device)
        
        if training:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                     np.exp(-1. * self.steps_done / self.epsilon_decay)
            self.steps_done += 1
            
            if random.random() < epsilon:
                # Exploration: random action
                action = random.randint(0, 5)
            else:
                # Exploitation: policy-guided action
                with torch.no_grad():
                    policy_probs, _ = self.policy_net(state)
                    action = int(torch.multinomial(policy_probs, 1).item())
        else:
            # Evaluation mode: always use policy
            with torch.no_grad():
                policy_probs, _ = self.policy_net(state)
                action = int(torch.argmax(policy_probs).item())
        
        self.action_counts[action] += 1
        return action
    
    def apply_tentative(self, action: int, state: torch.Tensor, mini_loader: Any) -> Tuple[bool, Dict]:
        """Enhanced tentative application with device-aware operations."""
        # Ensure state is on correct device
        state = state.to(self.device)
        
        # Save model state with device consistency
        prev_model_state = self._save_model_state()
        prev_metrics = self.evaluate(mini_loader)
        
        # Apply action with device management
        success, action_info = self._apply_action_safe(action)
        
        if not success:
            return False, {'reason': 'action_failed', 'error': action_info.get('error', 'Unknown')}
        
        # Evaluate new model
        new_metrics = self.evaluate(mini_loader)
        
        # Compute reward
        reward = self.reward_computer.compute_reward(prev_metrics, new_metrics, action)
        
        # Create experience for replay buffer with device consistency
        next_state = self.get_state(new_metrics)
        experience = Experience(
            state=state.cpu(),  # Store on CPU to save GPU memory
            action=action,
            reward=reward,
            next_state=next_state.cpu(),
            done=False,
            metrics_before=prev_metrics,
            metrics_after=new_metrics
        )
        self.experience_buffer.push(experience)
        
        # Decision: keep or rollback
        keep_change = reward > -0.1  # More lenient threshold for exploration
        
        if not keep_change:
            # Rollback with device consistency
            self._restore_model_state(prev_model_state)
            return False, {'reason': 'performance_degraded', 'reward': reward}
        
        # Keep change and update policy
        self._update_policy()
        
        # Record evolution step
        evolution_step = EvolutionStep(
            epoch=len(self.history),
            action=f"action_{action}",
            target_layers=action_info.get('target_layers', []),
            metrics_before=prev_metrics,
            metrics_after=new_metrics,
            info_gain=reward,
            parameters_before=prev_metrics.get('params', 0),
            parameters_after=new_metrics.get('params', 0)
        )
        self.history.append(evolution_step)
        
        return True, {
            'reward': reward,
            'action_info': action_info,
            'metrics_improvement': {
                k: new_metrics.get(k, 0) - prev_metrics.get(k, 0) 
                for k in ['accuracy', 'params', 'mutual_info']
            }
        }
    
    def _save_model_state(self) -> Dict[str, torch.Tensor]:
        """Save model state with proper device handling."""
        return {k: v.clone().detach() for k, v in self.base_model.state_dict().items()}
    
    def _restore_model_state(self, state_dict: Dict[str, torch.Tensor]):
        """Restore model state with device consistency."""
        # Ensure all tensors are on the correct device
        device_state_dict = {}
        for k, v in state_dict.items():
            device_state_dict[k] = v.to(self.device)
        
        self.base_model.load_state_dict(device_state_dict)
    
    def _apply_action_safe(self, action: int) -> Tuple[bool, Dict]:
        """Apply action with comprehensive error handling and device management."""
        action_map = {
            0: 'prune',
            1: 'expand', 
            2: 'mutate',
            3: 'branch',
            4: 'attention',
            5: 'hybrid'
        }
        
        try:
            if action < len(self.operators.operators):
                # Pass device information to operators
                metrics = {'device': self.device}
                modified_model, info = self.operators.operators[action].apply(self.base_model, metrics)
                
                # Ensure the modified model is on the correct device
                if modified_model != self.base_model:
                    modified_model.to(self.device)
                    # Update base_model reference
                    self.base_model = modified_model
                
                return True, {'action_type': action_map.get(action, 'unknown'), **info}
            else:
                return False, {'error': 'invalid_action'}
        except Exception as e:
            return False, {'error': f"Action failed: {str(e)}"}
    
    def _update_policy(self):
        """Update policy using PPO-style algorithm with device management."""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        try:
            # Sample batch from experience buffer
            experiences = self.experience_buffer.sample(self.batch_size)
            
            # Move tensors to device
            states = torch.stack([exp.state.to(self.device) for exp in experiences])
            actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long, device=self.device)
            rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32, device=self.device)
            next_states = torch.stack([exp.next_state.to(self.device) for exp in experiences])
            
            # Compute advantages using GAE (Generalized Advantage Estimation)
            with torch.no_grad():
                _, values = self.policy_net(states)
                _, next_values = self.policy_net(next_states)
                
                # Simple advantage computation (can be enhanced with GAE)
                advantages = rewards + self.gamma * next_values.squeeze() - values.squeeze()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy update
            policy_probs, current_values = self.policy_net(states)
            old_policy_probs = policy_probs.detach()
            
            # Policy loss (PPO clipped objective)
            action_probs = policy_probs.gather(1, actions.unsqueeze(1)).squeeze()
            old_action_probs = old_policy_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            ratio = action_probs / (old_action_probs + 1e-8)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Value loss
            value_loss = F.mse_loss(current_values.squeeze(), rewards + self.gamma * next_values.squeeze())
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.value_optimizer.step()
            
            # Update target network periodically
            if self.steps_done % 100 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
        except Exception as e:
            print(f"Policy update failed: {e}")
    
    def evaluate(self, loader) -> Dict:
        """Enhanced evaluation with device management."""
        self.base_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                if batch_idx >= 5:  # Quick evaluation on first 5 batches
                    break
                
                # Handle different loader formats
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, target = batch_data
                    data, target = data.to(self.device), target.to(self.device)
                else:
                    # Handle mini_loader_data format from trainer
                    continue
                    
                try:
                    output = self.base_model(data)
                    loss = F.cross_entropy(output, target)
                    total_loss += loss.item()
                    
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                except Exception as e:
                    print(f"Evaluation batch failed: {e}")
                    continue
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / min(5, len(loader)) if total > 0 else 1.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'params': sum(p.numel() for p in self.base_model.parameters()),
            'mutual_info': random.random(),  # Placeholder - integrate with IB
            'redundancy': random.random(),   # Placeholder
            'complexity': sum(p.numel() for p in self.base_model.parameters()) / 1e6,
            'entropy': random.random()       # Placeholder
        }
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'total_steps': self.steps_done,
            'current_epsilon': self.epsilon_end + (self.epsilon_start - self.epsilon_end) * 
                              np.exp(-1. * self.steps_done / self.epsilon_decay),
            'action_distribution': self.action_counts,
            'pareto_front_size': len(self.reward_computer.pareto_front),
            'experience_buffer_size': len(self.experience_buffer),
            'evolution_steps': len(self.history),
            'device': str(self.device)
        }

# Device-aware operators for robust architecture modifications
class DeviceAwareAttentionOperator(StructuralOperator):
    """Add attention mechanisms with proper device management."""
    def __init__(self, device: torch.device):
        self.device = device
        
    def apply(self, model, metrics):
        """Add attention mechanism with device consistency."""
        device = metrics.get('device', self.device)
        
        # For now, return model unchanged to avoid complex modifications
        # In a full implementation, this would add attention layers
        return model, {'action': 'attention', 'target_layers': ['attention_module'], 'device': str(device)}
    
    def can_apply(self, model, metrics):
        return True

class DeviceAwareHybridOperator(StructuralOperator):
    """Hybrid operations with device management."""
    def __init__(self, device: torch.device):
        self.device = device
        
    def apply(self, model, metrics):
        """Hybrid operation with device consistency."""
        device = metrics.get('device', self.device)
        
        # For now, return model unchanged to avoid complex modifications
        return model, {'action': 'hybrid', 'target_layers': ['hybrid_module'], 'device': str(device)}
    
    def can_apply(self, model, metrics):
        return True

class DeviceAwareBranchOperator(StructuralOperator):
    """Enhanced branching operator with device management."""
    def __init__(self, device: torch.device):
        self.device = device
        
    def apply(self, model, metrics):
        """Add branch with device consistency."""
        device = metrics.get('device', self.device)
        
        # For now, return model unchanged to avoid complex modifications
        # In a full implementation, this would add new branches
        return model, {'action': 'branch', 'target_layers': ['new_branch'], 'device': str(device)}
    
    def can_apply(self, model, metrics):
        return True 