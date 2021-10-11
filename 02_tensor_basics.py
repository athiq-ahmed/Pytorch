import numpy as np
import torch

x = torch.rand(3)
print(x)

print(torch.cuda.is_available())
print(torch.__version__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(device)

x = torch.empty(2, 3)
x = torch.rand(2, 3)
x = torch.ones(2, 3)
x = torch.zeros(2, 3, dtype=torch.int)
x = torch.tensor([2, 3, 4])
print(x)
print(x.dtype)
print(x.shape)


# Basic operations
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x + y  # Method 1
z = torch.add(x, y)  # Method 2
y.add_(x)  # Method 3 # inplace
z = x * y
z = x - y
z = x / y
print(z)


x = torch.rand(5, 3)
print(x)
print(x[:, 0])
print(x[1, 1].item())

x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)
z = x.view(-1)
print(z)
z = x.view(-1, 8)
print(z)


a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
print(b.dtype)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    print(z)

x = torch.ones(5, requires_grad=True)
print(x)
