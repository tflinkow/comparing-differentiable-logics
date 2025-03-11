from base.backends import Backend
from base.logic import Logic

class FuzzyLogic(Logic):
    def __init__(self, backend: Backend, abbrv: str, name: str, display_name: str = None):
        super().__init__(backend, abbrv, name, display_name)

    def LEQ(self, x, y):
        return 1. - self.backend.safe_div(self.backend.clamp_min(x - y, 0.), (self.backend.abs(x) + self.backend.abs(y)))

    def NOT(self, x):
        return 1. - x

class GoedelFuzzyLogic(FuzzyLogic):
    def __init__(self, backend: Backend, abbrv='GD', name='Goedel', display_name='Gödel'):
        super().__init__(backend, abbrv, name, display_name)

    def AND(self, x, y):
        return self.backend.minimum(x, y)

    def OR(self, x, y):
        return self.backend.maximum(x, y)

    def IMPL(self, x, y):
        return self.backend.where(x < y, 1., y)

class KleeneDienesFuzzyLogic(GoedelFuzzyLogic):
    def __init__(self, backend: Backend):
        super().__init__(backend, abbrv='KD', name='KleeneDienes', display_name='Kleene-Dienes')

    def IMPL(self, x, y):
        return Logic.IMPL(self, x, y)

class LukasiewiczFuzzyLogic(GoedelFuzzyLogic):
    def __init__(self, backend: Backend):
        super().__init__(backend, abbrv='LK', name='Lukasiewicz', display_name='Łukasiewicz')

    def AND(self, x, y):
        return self.backend.maximum(self.backend.zeros_like(x), x + y - 1.)

    def OR(self, x, y):
        return self.backend.minimum(self.backend.ones_like(x), x + y)

    def IMPL(self, x, y):
        return Logic.IMPL(self, x, y)

class ReichenbachFuzzyLogic(FuzzyLogic):
    def __init__(self, backend: Backend, abbrv='RC', name='Reichenbach', display_name='Reichenbach'):
        super().__init__(backend, abbrv=abbrv, name=name, display_name=display_name)

    def AND(self, x, y):
        return x * y

    def OR(self, x, y):
        return x + y - x * y

class GoguenFuzzyLogic(ReichenbachFuzzyLogic):
    def __init__(self, backend: Backend):
        super().__init__(backend, abbrv='GG', name='Goguen', display_name='Goguen')

    def IMPL(self, x, y):
        return self.backend.where(self.backend.logical_or(x <= y, x == 0.), self.backend.symbol(1.), self.backend.safe_div(y, x))

class ReichenbachSigmoidalFuzzyLogic(ReichenbachFuzzyLogic):
    def __init__(self, backend: Backend, s=9.):
        super().__init__(backend, abbrv='RCS', name='ReichenbachSigmoidal', display_name=r'sig. Reichenbach')
        self.s = self.backend.ctor_param('s', s)

    def IMPL(self, x, y):
        exp = self.backend.exp(self.backend.symbol(self.s / 2))

        numerator = (1. + exp) * self.backend.sigmoid(self.s * super().IMPL(x, y) - self.s/2) - 1.
        denominator = exp - 1.

        I_s = self.backend.clamp_max(self.backend.safe_div(numerator, denominator), 1.)

        return I_s

class YagerFuzzyLogic(FuzzyLogic):
    def __init__(self, backend: Backend, p=2):
        super().__init__(backend, abbrv='YG', name='Yager')
        self.p = self.backend.ctor_param('p', p)

    def AND(self, x, y):
        return self.backend.clamp_min(1. - self.backend.pow( self.backend.pow(1. - x, self.p) + self.backend.pow(1. - y, self.p), 1. / self.p), 0.)

    def OR(self, x, y):
        return self.backend.clamp_max(self.backend.pow( self.backend.pow(x, self.p) + self.backend.pow(y, self.p), 1. / self.p) , 1.)

    def IMPL(self, x, y):
        return self.backend.where(self.backend.logical_and(x == 0., y == 0.), self.backend.ones_like(x), self.backend.pow(y, x))