from __future__ import print_function

from contextlib import contextmanager

import torch
import numpy as np

import copy

import sys
sys.path.append(".")
sys.path.append("..")

from base.logic import Logic
from base.backends import NumpyBackend

from .constraints import Constraint

@contextmanager
def maybe(context_manager, flag: bool):
    if flag:
        with context_manager as cm:
            yield cm
    else:
        yield None

# GradNorm (https://arxiv.org/abs/1711.02257) based on https://github.com/brianlan/pytorch-grad-norm/blob/master/train.py
class GradNorm():
    def __init__(self, device: torch.device, model: torch.nn.Module):
        self.model = model
        self.initial_loss = 0.
        self.device = device

    def weighted_loss(self, batch_index: int, ce_loss: torch.Tensor, dl_loss: torch.Tensor, alpha: float):
        task_loss = torch.stack([ce_loss, dl_loss])
        weighted_loss = torch.mul(self.model.loss_weights, task_loss)

        if batch_index == 1:
            self.initial_loss = task_loss.detach().cpu().numpy()

        total_loss = torch.sum(weighted_loss)
        total_loss.backward(retain_graph=True)

        self.model.loss_weights.grad = None

        W = self.model.last_shared_layer
        norms = torch.stack([torch.norm(torch.mul(self.model.loss_weights[i], torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)[0])) for i in range(2)])

        loss_ratio = NumpyBackend().safe_div(task_loss.detach().cpu().numpy(), self.initial_loss)
        inverse_train_rate = NumpyBackend().safe_div(loss_ratio, np.mean(loss_ratio))

        mean_norm = np.mean(norms.detach().cpu().numpy())
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False, device=self.device)
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

        self.model.loss_weights.grad = torch.autograd.grad(grad_norm_loss, self.model.loss_weights)[0]

        return total_loss

    def renormalise(self):
        normalise_coeff = 2 / torch.sum(self.model.loss_weights.data, dim=0)
        self.model.loss_weights.data *= normalise_coeff

        print(f'loss_weights={self.model.loss_weights.data}')

    def __enter__(self):
        return self

    def __exit__(self, _exc_type: None, _exc_value: None, _exc_traceback: None):
        self.renormalise()

# based on https://github.com/oscarknagg/adversarial/blob/master/adversarial/functional.py
class PGD:
    def __init__(self, device: torch.device, steps: int, mean: tuple[float, float, float], std: tuple[float, float, float], gamma: float = 64/255):
        self.device = device
        self.steps = steps
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)
        self.gamma = gamma

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
    
    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def random_perturbation(self, x: torch.tensor, eps: float):
        perturbation = torch.normal(torch.zeros_like(x), torch.ones_like(x))
        perturbation = torch.sign(perturbation) * eps

        return x + perturbation

    @torch.enable_grad
    def attack(self, model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor, logic: Logic, constraint: Constraint, eps: float):
        model = copy.deepcopy(model)
        model.eval()

        adv = self.denormalise(inputs.clone().detach().requires_grad_(True).to(inputs.device))

        # random uniform start
        adv = self.random_perturbation(adv, eps)
        adv.requires_grad_(True)

        for _ in range(self.steps):
            _adv = adv.clone().detach().requires_grad_(True)

            loss, _ = constraint.eval(model, inputs, self.normalise(_adv), labels, logic, train=True, skip_sat=True)
            loss.backward()

            with torch.no_grad():
                gradients = _adv.grad.sign() * self.gamma
                adv += gradients

            # project back into l_norm ball and correct range
            adv = torch.max(torch.min(adv, inputs + eps), inputs - eps)
            adv = torch.clamp(adv, min=0, max=1)

        return self.normalise(adv.detach())