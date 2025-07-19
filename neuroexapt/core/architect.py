
"""
defgroup group_architect Architect
ingroup core
Architect module for NeuroExapt framework.
"""


import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):
    """
    Manages the architecture parameters (alphas) of the network.
    This class implements the bi-level optimization strategy for differentiable
    architecture search. It updates the architecture parameters by approximating
    the architectural gradient.
    """
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        self.criterion: Optional[nn.Module] = None  # Will be set during training
        
        # Add compatibility with SimpleArchitect interface
        self.arch_update_freq = getattr(args, 'arch_update_freq', 50)
        self.warmup_epochs = getattr(args, 'warmup_epochs', 5)
        self.current_epoch = 0
        self.step_count = 0
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
    
    def should_update_arch(self) -> bool:
        """判断是否应该更新架构"""
        # 预热期跳过
        if self.current_epoch < self.warmup_epochs:
            self.step_count += 1  # 仍然增加计数但不更新
            return False
        
        # 简单的频率控制
        self.step_count += 1
        should_update = self.step_count % self.arch_update_freq == 0
        return should_update

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        """
        Compute the forward pass for a single-step unrolled model.
        This creates a "virtual" model where the weights are updated by one step
        of gradient descent, and then computes the loss on this virtual model.
        This is the core of the gradient approximation in DARTS.
        """
        # Clear any existing gradients
        self.model.zero_grad()
        
        if self.criterion is None:
            raise ValueError("Criterion not set")
        loss = self.criterion(self.model(input), target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        
        # Create a virtual model with one-step updated weights
        unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        """
        Performs one step of architecture optimization.
        Args:
            input_train, target_train: data from the training set for inner optimization.
            input_valid, target_valid: data from the validation set for outer optimization.
            eta: learning rate for the inner optimization (network weights).
            network_optimizer: optimizer for the network weights.
            unrolled: whether to use the unrolled approximation for the gradient.
        """
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        if self.criterion is None:
            raise ValueError("Criterion not set")
        loss = self.criterion(self.model(input_valid), target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        """
        Computes the architectural gradient using the unrolled approximation.
        This involves a backward pass through the one-step virtual model update.
        """
        # Create the virtual model
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        # Compute the loss on the virtual model using validation data
        if self.criterion is None:
            raise ValueError("Criterion not set")
        unrolled_loss = self.criterion(unrolled_model(input_valid), target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        
        # Manually compute the second-order gradient term
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        """
        Constructs a new model instance and loads the given weights (theta) into it.
        """
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target):
        """Efficient Hessian-vector product via automatic differentiation.

        This version avoids the two extra forward passes of the finite-difference
        approximation used previously, reducing both compute time and memory
        pressure. The algorithm follows:

            1. Compute first-order gradients of the loss w.r.t. *weights* with
               ``create_graph=True`` so that a subsequent backward pass can
               construct the Hessian.
            2. Form the dot-product of these gradients and the supplied vector.
            3. Take gradients of the dot-product w.r.t. *architecture* params to
               obtain the Hessian-vector product.
        """

        if self.criterion is None:
            raise ValueError("Criterion not set")

        # Forward + first backward pass to obtain gradients wrt network weights
        loss = self.criterion(self.model(input), target)
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

        # Dot product between gradient list and the provided vector
        # Ensure we start accumulation with a tensor, not a Python int, to satisfy type checkers
        grad_dot = torch.zeros(1, device=loss.device, dtype=loss.dtype)
        for g, v in zip(grads, vector):
            grad_dot = grad_dot + (g * v).sum()

        # Second backward: gradients of dot product wrt architecture parameters
        hvp = torch.autograd.grad(grad_dot, self.model.arch_parameters(), retain_graph=False)

        # Detach to avoid bloating the autograd graph in the caller
        return [h.detach() for h in hvp]

    def cleanup_gradients(self):
        """清理梯度以防止内存泄漏（安全版本）"""
        try:
            # 清理主模型梯度
            self.model.zero_grad()
            
            # 清理架构参数梯度
            for param in self.model.arch_parameters():
                if param.grad is not None:
                    param.grad.data.zero_()
            
            # 清理优化器状态（如果需要）
            if hasattr(self.optimizer, 'state') and len(self.optimizer.state) > 100:
                self.optimizer.state.clear()
            
        except Exception as e:
            # 静默处理清理错误，避免影响训练
            pass
        
        # 不调用torch.cuda.empty_cache()以避免死锁
        # 让PyTorch自动管理GPU内存 