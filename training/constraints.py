import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Callable

from functools import reduce

import sys

sys.path.append(".")
sys.path.append("..")

from .group_definitions import Class

from base.backends import TorchBackend

from base.logic import Logic
from base.boolean_logic import BooleanLogic
from base.dl2 import DL2
from base.fuzzy_logics import FuzzyLogic

class Constraint(ABC):
    def __init__(self, device: torch.device, eps: float):
        self.device = device
        self.boolean_logic = BooleanLogic(TorchBackend())

        assert 0. <= eps <= 1., 'eps should be within the range [0, 1]'
        self.eps = torch.tensor(eps, device=self.device)

    @abstractmethod
    def get_constraint(self, model: torch.nn.Module, inputs: torch.Tensor | None, adv: torch.Tensor | None, labels: torch.Tensor | None) -> Callable[[Logic], torch.Tensor]:
        pass

    # usage:
    # loss, sat = eval()
    # sat indicates whether the constraint is satisfied or not
    def eval(self, model: torch.nn.Module, inputs: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor, logic: Logic, reduction: str | None = None, skip_sat: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        constraint = self.get_constraint(model, inputs, adv, labels)
        loss, sat = None, None

        loss = constraint(logic)
        assert not torch.isnan(loss).any()

        if isinstance(logic, FuzzyLogic):
            loss = torch.ones_like(loss) - loss

        if not skip_sat:
            sat = constraint(self.boolean_logic).float()

        def agg(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None

            if reduction == None:
                return value
            elif reduction == 'mean':
                return torch.mean(value)
            elif reduction == 'sum':
                return torch.sum(value)
            else:
                assert False, f'unsupported reduction {reduction}'

        return agg(loss), agg(sat)

class StandardRobustnessConstraint(Constraint):
    def __init__(self, device: torch.device, eps: float, delta: float):
        super().__init__(device, eps)

        assert 0. <= delta <= 1., 'delta is a probability and should be within the range [0, 1]'
        self.delta = torch.tensor(delta, device=self.device)

    def get_constraint(self, model: torch.nn.Module, inputs: torch.Tensor, adv: torch.Tensor, _labels: None) -> Callable[[Logic], torch.tensor]:
        # combine two separate forward passes (i.e. model(inputs) and model(adv) )
        outputs = model(torch.cat([inputs, adv], dim=0))

        outputs_inputs, outputs_adv = outputs.chunk(2, dim=0)
        diff = F.softmax(outputs_adv, dim=1) - F.softmax(outputs_inputs, dim=1)

        return lambda l: l.LEQ(torch.linalg.vector_norm(diff, ord=float('inf'), dim=1), self.delta)

class StrongClassificationRobustnessConstraint(Constraint):
    def __init__(self, device: torch.device, eps: float, delta: float):
        super().__init__(device, eps)
        self.delta = torch.tensor(delta, device=self.device)

    def get_constraint(self, model: torch.nn.Module, _inputs: None, adv: torch.Tensor, labels: torch.Tensor) -> Callable[[Logic], torch.Tensor]:
        out = model(adv)[torch.arange(len(labels)), labels]
        return lambda l: l.LEQ(self.delta, out)

class EvenOddConstraint(Constraint):
    def __init__(self, device: torch.device, eps: float, delta: float = .6, gamma: float = 3.):
        super().__init__(device, eps)

        assert 0. <= delta <= 1., 'delta is a probability and should be within the range [0, 1]'
        self.delta = torch.tensor(delta, device=self.device)
        self.gamma = torch.tensor(gamma, device=device)

    def get_vacuously_true(self, model: torch.nn.Module, adv: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(model(adv), dim=1)

        even = torch.sum(probs[:, [0, 2, 4, 6, 8]], dim=1)
        odd = torch.sum(probs[:, [1, 3, 5, 7, 9]], dim=1)

        print(f'even={even}, odd={odd}')

        return torch.tensor([(self.delta > A).sum() for A in [even, odd]], device=self.device)

    def get_constraint(self, model: torch.nn.Module, _inputs: None, adv: torch.Tensor, labels: torch.Tensor) -> Callable[[Logic], torch.Tensor]:
        out = model(adv)
        probs = F.softmax(out, dim=1)

        even = torch.sum(probs[:, [0, 2, 4, 6, 8]], dim=1)
        odd = torch.sum(probs[:, [1, 3, 5, 7, 9]], dim=1)

        even_without_true = even - torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)
        odd_without_true = odd - torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)

        # referring to the true label here which might not necessarily be the prediction the network makes (but it most likely is, because it's MNIST)
        return lambda l: (
            reduce(l.AND,
            [
                # DL2 doesn't have negation, thus push negation inwards and hard-code material implication
                l.OR(l.LT(A, self.delta), l.LEQ(self.gamma * B, C)) for A, B, C in [(even, odd, even_without_true), (odd, even, odd_without_true)]
            ]) if isinstance(l, DL2) else
            reduce(l.AND,
            [
                l.IMPL(l.LEQ(self.delta, A), l.LEQ(self.gamma * B, C)) for A, B, C in [(even, odd, even_without_true), (odd, even, odd_without_true)]
            ])
        )

class GroupConstraint(Constraint):
    def __init__(self, device: torch.device, eps: float, delta: float, groups: list[list[Class]]):
        super().__init__(device, eps)
        self.group_indices: list[list[int]] = [[x.class_index for x in group] for group in groups]

        assert 0. <= delta <= 1., 'delta is a probability and should be within the range [0, 1]'
        self.delta = torch.tensor(delta, device=self.device)

    def get_constraint(self, model: torch.nn.Module, _inputs: None, adv: torch.Tensor, _labels: None) -> Callable[[Logic], torch.Tensor]:
        probs = F.softmax(model(adv), dim=1)
        sums = [torch.sum(probs[:, indices], dim=1) for indices in self.group_indices]

        return lambda l: reduce(l.AND,
            [
                l.OR(
                    l.LEQ(s, self.delta),
                    l.LEQ(1. - self.delta, s)
                )
                for s in sums
            ]
        )

class ClassSimilarityConstraint(Constraint):
    def __init__(self, device: torch.device, eps: float, indices, delta: float = .1):
        super().__init__(device, eps)
        self.indices = indices

        assert 0. <= delta <= 1., 'delta is a probability and should be within the range [0, 1]'
        self.delta = torch.tensor(delta, device=self.device)

    def get_vacuously_true(self, model: torch.nn.Module, adv: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(model(adv), dim=1)

        return torch.tensor([(self.delta > probs[:, i[0]]).sum() for i in self.indices], device=self.device)

    def get_constraint(self, model: torch.nn.Module, _inputs: None, adv: torch.Tensor, _labels: None) -> Callable[[Logic], torch.Tensor]:
        probs = F.softmax( model(adv), dim=1)

        return lambda l: (
            reduce(l.AND, 
                [
                    # DL2 doesn't have negation, thus push negation inwards and hard-code material implication
                    l.OR(
                        l.LT(probs[:, i[0]], self.delta),
                        l.LEQ(probs[:, i[2]], probs[:, i[1]])
                    )
                    for i in self.indices
                ]
            ) if isinstance(l, DL2) else 
            reduce(l.AND,
                [
                    l.IMPL(
                        l.LEQ(self.delta, probs[:, i[0]]),
                        l.LEQ(probs[:, i[2]], probs[:, i[1]])
                    )
                    for i in self.indices
                ]
            )
        )