import torch

a = torch.tensor([5.], requires_grad=True); print(a)
b = torch.tensor([6.], requires_grad=True); print(b)
y = a**3 - b**2; print(y)

# dy/da = 3*a**2=75
# dy/db = -2b= -12

y.backward()
print(a.grad)
print(b.grad)

w = torch.randn(10, 1, requires_grad=True); print(w)
b = torch.randn(1, requires_grad=True);print(b)
x = torch.randn(1, 10);print(x)

output = torch.matmul(x, w) + b; print(output)
loss = 1 - output; print(loss)
loss.backward()
w.grad

with torch.no_grad():
    w = w - 0.001 * w.grad.data