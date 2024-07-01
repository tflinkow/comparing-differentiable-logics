from base.backends import Backend
from base.logic import Logic
    
class DL2(Logic):
    def __init__(self, backend: Backend):
        super().__init__(backend, 'DL2', 'DL2')

    def LEQ(self, x, y):
        return self.backend.clamp_min(x - y, 0.0)

    def NOT(self, x):
        # technically, negation is not supported in DL2, but this allows to use base class implication definition
        return 1.0 - x

    def AND(self, x, y):
        return x + y

    def OR(self, x, y):
        return x * y