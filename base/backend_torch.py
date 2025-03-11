import torch

from base.backends import Backend

class TorchBackend(Backend):
    def minimum(self, x, y):
        return torch.minimum(x, y)

    def maximum(self, x, y):
        return torch.maximum(x, y)

    def where(self, condition, x, y):
        return torch.where(condition, x, y)

    def zeros_like(self, x):
        return torch.zeros_like(x)

    def ones_like(self, x):
        return torch.ones_like(x)

    def symbol(self, x):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        return torch.tensor(x, device=device)

    def ctor_param(self, _name: None, value):
        return value

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def clamp_max(self, x, y):
        return torch.clamp(x, max=y)

    def clamp_min(self, x, y):
        return torch.clamp(x, min=y)

    def abs(self, x):
        return torch.abs(x)

    def exp(self, x):
        return torch.exp(x)

    def pow(self, x, y):
        return torch.pow(self.safe_zero(x), y)

    def logical_and(self, x, y):
        return torch.logical_and(x, y)

    def logical_or(self, x, y):
        return torch.logical_or(x, y)

    def logical_not(self, x):
        return torch.logical_not(x)
    
    def safe_div(self, x, y):
        return x / torch.where(y == 0., torch.finfo(y.dtype).eps, y)

    def safe_zero(self, x):
        return torch.where(x == 0., torch.full_like(x, torch.finfo(x.dtype).eps), x)