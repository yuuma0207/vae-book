import torch

x = torch.tensor([1,2,3,4])
output = torch.cumprod(x, dim=0)
print(output)