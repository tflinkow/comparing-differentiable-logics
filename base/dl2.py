from base.backends import Backend
from base.logic import Logic
    
class DL2(Logic):
    def __init__(self, backend: Backend):
        super().__init__(backend, 'DL2', 'DL2')

    def LEQ(self, x, y):
        return self.backend.clamp_min(x - y, 0.)

    def LT(self, x, y):
        xi = self.backend.symbol(1.)
        return self.AND(self.LEQ(x, y), xi * (x == y).float())

    def NOT(self, _x: None):
        raise NotImplementedError('DL2 does not have general negation - rewrite the constraint to push negation inwards, e.g. NOT(x <= y) should be (y < x)')

    def AND(self, x, y):
        return x + y

    def OR(self, x, y):
        return x * y