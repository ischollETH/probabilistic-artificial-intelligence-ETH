import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[7],[8]])

print(a)
print(b)
print(a*b)
print(torch.mul(a,b))
