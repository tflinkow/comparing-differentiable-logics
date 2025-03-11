import sympy

from base.backends import Backend

class SympyBackend(Backend):
    def minimum(self, x, y):
        return sympy.Min(x, y)

    def maximum(self, x, y):
        return sympy.Max(x, y)

    def where(self, condition, x, y):
        return sympy.Piecewise((x, condition), (y, True))

    def zeros_like(self, _x: None):
        return 0

    def ones_like(self, _x: None):
        return 1

    def symbol(self, x):
        return x

    def ctor_param(self, name, _value: None):
        return sympy.Symbol(name)

    def sigmoid(self, x):
        return 1 / (1 + sympy.exp(-x))

    def clamp_max(self, x, y):
        return sympy.Max(x, y)

    def clamp_min(self, x, y):
        return sympy.Min(x, y)

    def abs(self, x):
        return sympy.Abs(x)

    def exp(self, x):
        return sympy.exp(x)

    def pow(self, x, y):
        return x**y

    def logical_and(self, x, y):
        return sympy.And(x, y)

    def logical_or(self, x, y):
        return sympy.Or(x, y)

    def logical_not(self, x):
        return sympy.Not(x)

    def safe_div(self, x, y):
        return x / y

    def safe_zero(self, x):
        return x