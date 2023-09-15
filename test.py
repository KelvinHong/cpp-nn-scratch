"""
Playground for testing relevant PyTorch codes.
"""

import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.data)
print(x.grad)
print(x.grad_fn)

y = 2+ x.sum()
print(y)
print(y.grad_fn)
print(y.grad_fn.next_functions)
