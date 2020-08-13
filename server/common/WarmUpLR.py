# -*- coding: utf-8 -*-
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """
    Warm-up is a way to reduce the primacy effect of the early training examples.
    Without it, you may need to run a few extra epochs to get the convergence desired,
    as the model un-trains those early superstitions.
    ref: https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
    Args:
        optimizer: optimizer (e.g. SGD)
        total_iters: total iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """use the first m mini-batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
