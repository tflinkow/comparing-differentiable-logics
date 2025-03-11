from abc import ABC, abstractmethod
from typing import Any

class Backend(ABC):
    @abstractmethod
    def minimum(self, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def maximum(self, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def where(self, condition, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def zeros_like(self, x: Any) -> Any:
        pass

    @abstractmethod
    def ones_like(self, x: Any) -> Any:
        pass

    @abstractmethod
    def symbol(self, x: Any) -> Any:
        pass

    @abstractmethod
    def ctor_param(self, name: str, value: Any) -> Any:
        pass

    @abstractmethod
    def sigmoid(self, x: Any) -> Any:
        pass

    @abstractmethod
    def clamp_max(self, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def clamp_min(self, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def abs(self, x: Any) -> Any:
        pass

    @abstractmethod
    def exp(self, x: Any) -> Any:
        pass

    @abstractmethod
    def pow(self, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def logical_and(self, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def logical_or(self, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def logical_not(self, x: Any) -> Any:
        pass

    @abstractmethod
    def safe_div(self, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def safe_zero(self, x: Any) -> Any:
        pass

from base.backend_sympy import SympyBackend
from base.backend_numpy import NumpyBackend
from base.backend_torch import TorchBackend