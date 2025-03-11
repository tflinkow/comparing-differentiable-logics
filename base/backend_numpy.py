import numpy

from base.backends import Backend

class NumpyBackend(Backend):
    def minimum(self, x, y):
        return numpy.minimum(x, y)

    def maximum(self, x, y):
        return numpy.maximum(x, y)

    def where(self, condition, x, y):
        return numpy.where(condition, x, y)

    def zeros_like(self, _x: None):
        return 0.

    def ones_like(self, _x: None):
        return 1.

    def symbol(self, x):
        return x

    def ctor_param(self, _name: None, value):
        return value

    def sigmoid(self, x):
        return 1. / (1. + numpy.exp(-x))

    def clamp_max(self, x, y):
        return numpy.clip(x, a_min=None, a_max=y)

    def clamp_min(self, x, y):
        return numpy.clip(x, a_min=y, a_max=None)

    def abs(self, x):
        return numpy.abs(x)

    def exp(self, x):
        return numpy.exp(x)

    def pow(self, x, y):
        return numpy.power(x, y)

    def logical_and(self, x, y):
        return x & y

    def logical_or(self, x, y):
        return x | y

    def logical_not(self, x):
        return ~x

    def safe_div(self, x, y):
        y = numpy.asarray(y)
        return x / numpy.where(y == 0., numpy.finfo(y.dtype).eps, y)

    def safe_zero(self, x):
        return x