from base.backends import Backend
from base.logic import Logic

class BooleanLogic(Logic):
    def __init__(self, backend: Backend):
        super().__init__(backend, 'unused', 'Boolean')

    def LEQ(self, x, y):
        return x <= y

    def NOT(self, x):
        return self.backend.logical_not(x)

    def AND(self, x, y):
        return self.backend.logical_and(x, y)

    def OR(self, x, y):
        return self.backend.logical_or(x, y)

    def IMPL(self, x, y):
        return self.backend.logical_or(self.backend.logical_not(x), y)