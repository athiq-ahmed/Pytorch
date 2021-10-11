import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * 2
z = z.mean()
print(z)

z.backward()
print(x.grad)

v = torch.tensor([0.485, 0.456, 0.406], dtype=float)
z.backward(v)
print(z)

x = torch.randn(3, requires_grad=True)
print(x)
x.requires_grad_(False)
print(x)

y = x.detach()
print(y)

with torch.no_grad():
    y = x + 2
    print(y)


weights = torch.ones(4, requires_grad=True)
for epoch in range(2):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()
