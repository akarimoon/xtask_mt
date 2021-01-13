import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random
from itertools import accumulate

class PCGrad():
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad()

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        input:
        - objectives: a list of objectives
        '''
        grads, shapes, numel = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads)
        self._set_grad(pc_grad, shapes, numel)

    def _proj_grad(self, grad):
        for k in range(self.num_tasks):
            inner_product = torch.sum(grad * self.grads[k])
            proj_direction = inner_product / (torch.sum(self.grads[k] * self.grads[k]) + 1e-12)
            grad -= torch.min(proj_direction, torch.zeros_like(proj_direction)) * self.grads[k]
        return grad

    def _project_conflicting(self, grads, shapes=None):
        self.num_tasks = len(grads)
        random.shuffle(grads)

        # gradient projection
        self.grads = torch.stack(grads, dim=0)  # (T, # of params)
        pc_grad = self.grads.clone()

        pc_grad = torch.sum(torch.stack(list(map(self._proj_grad, list(pc_grad)))), dim=0)  # (of params, )

        return pc_grad

    def _set_grad(self, grads, shapes, numel):
        indices = [0, ] + [v for v in accumulate(numel)]
        params = [p for group in self._optim.param_groups for p in group['params']]
        assert len(params) == len(shapes) == len(indices[:-1])
        for param, shape, start_idx, end_idx in zip(params, shapes, indices[:-1], indices[1:]):
            if shape is not None and param.grad is not None:
                param.grad[...] = grads[start_idx:end_idx].view(shape)  # copy proj grad

    def _pack_grad(self, objectives):
        grads = []
        shapes = [p.shape if p.requires_grad is True else None
                    for group in self._optim.param_groups for p in group['params']]
        numel = [p.numel() if p.requires_grad is True else 0
                    for group in self._optim.param_groups for p in group['params']]

        for obj in objectives:
            self._optim.zero_grad()
            obj.backward(retain_graph=True)
            devices = [p.device for group in self._optim.param_groups for p in group['params']]
            grad = self._retrieve_and_flatten_grad()
            grads.append(self._fill_zero_grad(grad, numel, devices))
        return grads, shapes, numel

    def _retrieve_and_flatten_grad(self):
        grad = [p.grad.clone().flatten() if (p.requires_grad is True and p.grad is not None)
                else None for group in self._optim.param_groups for p in group['params']]
        return grad

    def _fill_zero_grad(self, grad, numel, devices):
        return torch.cat([g if g is not None else torch.zeros(numel[i], device=devices[i]) for i, g in enumerate(grad)])