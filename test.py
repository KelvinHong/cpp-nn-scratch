"""
Playground for testing relevant PyTorch codes.
"""

import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + x
print(y.grad_fn)
print(y.grad_fn.next_functions)



