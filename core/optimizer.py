import torch.optim as optim


class Optimizer():
    def __init__(self, optimizer: str | optim.Optimizer, config: dict = {}, model_parameters = None):
        assert (isinstance(optimizer, str) and model_parameters is not None) or isinstance(optimizer, optim.Optimizer)
        self._config = config

        if isinstance(optimizer, str):
            self._optimizer = getattr(optim, optimizer)(model_parameters, **config)
        else:
            self._optimizer = optimizer


    def __repr__(self):
        return f"Optimizer: {self._optimizer}"


    def step(self):
        self._optimizer.step()


    def zero_grad(self):
        self._optimizer.zero_grad()