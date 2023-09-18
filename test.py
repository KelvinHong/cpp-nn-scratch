"""
Playground for testing relevant PyTorch codes.
"""

import torch

def recursive_nf(next_fs, level = 1):
    for i in range(len(next_fs)):
        print("===="*level + str(level) + ": " + next_fs[i][0].__class__.__name__)
        if next_fs[i][0] is not None:
            recursive_nf(next_fs[i][0].next_functions, level=level+1)
    


def visualize_node(loss: torch.Tensor):
    print("Start visualizing...")
    print(loss.grad_fn)
    print(loss.grad_fn.__class__.__name__)
    recursive_nf(loss.grad_fn.next_functions)
    print("End of visualization.")

x = torch.rand((5,3), requires_grad=True)
fc1 = torch.nn.Linear(3, 4)
fc2 = torch.nn.Linear(4, 6)
x1 = fc1(x)
x2 = torch.nn.ReLU()(x1)
y = fc2(x2)
L = y.sum()
visualize_node(L)
# print(L.grad_fn)
# print(L.grad_fn.next_functions)
# print(L.grad_fn.next_functions[0][0].next_functions)
# print(L.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
# print(L.grad_fn.next_functions[0][0].next_functions[1][0].next_functions)

# x = torch.tensor([1.,2,3], requires_grad=True)
# w = torch.tensor([0.1,0.2,0.1], requires_grad=True)
# y = w * x
# z = y.sum()

# print("z:")
# print("\tgrad_fn", z.grad_fn)
# print("\tgrad_fn.metadata", z.grad_fn.metadata)
# print("\tgrad_fn.nf", z.grad_fn.next_functions)
# print("\tgrad_fn.nf.nf", z.grad_fn.next_functions[0][0].next_functions)
# print("\tgrad_fn.nf.metadata", z.grad_fn.next_functions[0][0].metadata)

# z.backward()
# print(x.grad)
# print(w.grad)

