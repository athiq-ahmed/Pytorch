"""
Using Numpy 
"""

import numpy as np

# f = w * x
# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0

# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_predicted):
    return ((y - y_predicted) ** 2).mean()


# gradient
# MSE = 1 / N * (y_pred - y) ** 2
# dJ/dw = 1/N 2x(y_pred - y)
def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()


print(f"Prediction before training: f(5)={forward(5):.3f}")

# Training
learning_rate = 0.01
n_iterations = 15

for epoch in range(n_iterations):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    dw = gradient(X, Y, y_pred)
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}:w = {w:.3f}, loss={l:.8f}")

print(f"Prediction After training: f(5)={forward(5):.3f}")


"""
Using PyTorch
"""

import torch

# f = w * x
# f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_predicted):
    return ((y - y_predicted) ** 2).mean()


# gradient
# MSE = 1 / N * (y_pred - y) ** 2
# dJ/dw = 1/N 2x(y_pred - y)
# def gradient(x, y, y_predicted):
#     return np.dot(2 * x, y_predicted - y).mean()


print(f"Prediction before training: f(5)={forward(5):.3f}")

# Training
learning_rate = 0.01
n_iterations = 15

for epoch in range(n_iterations):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    l.backward()  # dw = gradient(X, Y, y_pred)

    with torch.no_grad():
        w -= learning_rate * w.grad

    w.grad.zero_()

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}:w = {w:.3f}, loss={l:.8f}")

print(f"Prediction After training: f(5)={forward(5):.3f}")
