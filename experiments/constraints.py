import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Callable

from functools import reduce

import sys

from base.boolean_logic import Logic
sys.path.append(".")
sys.path.append("..")

from .group_definitions import Class

from base.backends import TorchBackend
from base.boolean_logic import *
from base.fuzzy_logics import FuzzyLogic

class Constraint(ABC):
    def __init__(self, eps: torch.Tensor):
        self.eps = eps
        self.boolean_logic = BooleanLogic(TorchBackend())

    def get_probabilities(self, outputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(outputs, dim=1)

    @abstractmethod
    def get_constraint(self, model: torch.nn.Module, inputs: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor) -> Callable[[Logic], torch.Tensor]:
        pass

    # usage:
    # loss, sat = eval()
    # where sat returns whether the constraint is satisfied or not
    def eval(self, model: torch.nn.Module, inputs: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor, logic: Logic, train: bool, skip_sat: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        constraint = self.get_constraint(model, inputs, adv, labels)

        loss = constraint(logic)
        sat = constraint(self.boolean_logic).float() if not skip_sat else None

        assert not torch.isnan(loss).any()

        if isinstance(logic, FuzzyLogic):
            loss = torch.ones_like(loss) - loss

        if train:
            return torch.mean(loss), torch.mean(sat) if not skip_sat else None
        else:
            return torch.sum(loss), torch.sum(sat) if not skip_sat else None

class RobustnessConstraint(Constraint):
    def __init__(self, eps: torch.Tensor, delta: torch.Tensor):
        super().__init__(eps)
        self.delta = delta

    def get_constraint(self, model: torch.nn.Module, inputs: torch.Tensor, adv: torch.Tensor, _labels: None) -> Callable[[Logic], torch.tensor]:
        diff = F.softmax(model(adv), dim=1) - F.softmax(model(inputs), dim=1)

        return lambda l: l.LEQ(torch.linalg.vector_norm(diff, ord=float('inf'), dim=1), self.delta)

class GroupConstraint(Constraint):
    def __init__(self, eps: torch.tensor, delta: torch.Tensor, groups: list[list[Class]]):
        super().__init__(eps)
        self.delta = delta
        self.group_indices: list[list[int]] = [[x.class_index for x in group] for group in groups]

    def get_constraint(self, model: torch.nn.Module, _inputs: None, adv: torch.Tensor, _labels: None) -> Callable[[Logic], torch.tensor]:
        probs = F.softmax(model(adv), dim=1)
        sums = [torch.sum(probs[:, indices], dim=1) for indices in self.group_indices]

        return lambda l: reduce(l.AND, [l.OR(l.LEQ(s, self.delta), l.LEQ(1. - self.delta, s)) for s in sums])

class ClassSimilarityConstraint(Constraint):
    def __init__(self, eps: torch.tensor, indices):
        super().__init__(eps)
        self.indices = indices

    def get_constraint(self, model: torch.nn.Module, _inputs: None, adv: torch.Tensor, _labels: None) -> Callable[[Logic], torch.tensor]:
        probs = F.softmax(model(adv), dim=1)

        return lambda l: reduce(l.AND, [l.IMPL(l.LEQ(self.eps, probs[:, i[0]]), l.LEQ(probs[:, i[2]], probs[:, i[1]])) for i in self.indices])