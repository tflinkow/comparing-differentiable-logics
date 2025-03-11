from abc import ABC, abstractmethod
from typing import Any

from base.backends import Backend

class Logic(ABC):
    def __init__(self, backend: Backend, abbrv: str, name: str, display_name: str = None):
        self.backend = backend
        self.abbrv = abbrv
        self.name = name

        if display_name is not None:
            self.display_name = display_name
        else:
            self.display_name = self.name

    @abstractmethod
    def LEQ(self, x, y) -> Any:
        pass

    @abstractmethod
    def NOT(self, x) -> Any:
        pass

    @abstractmethod
    def AND(self, x, y) -> Any:
        pass

    @abstractmethod
    def OR(self, x, y) -> Any:
        pass

    def IMPL(self, x, y) -> Any:
        return self.OR(self.NOT(x), y)

    def EQUIV(self, P, Q) -> Any:
        return self.AND(self.IMPL(P, Q), self.IMPL(Q, P))