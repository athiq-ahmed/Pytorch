"""
1. Design model(input, output, forward)
2. Construct a loss and optimizer
3. Training loop
    1. forward pass: compute predictions
    2. backward pass: gradients
    3. update weights

"""

import torch
import torch.nn as nn

# f = w * x
# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)


class LinearReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearReg, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearReg(input_size, output_size)

print(f"Prediction before training: f(5)={model(X_test).item():.3f}")


# Training
learning_rate = 0.01
n_iterations = 100


loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(n_iterations):
    y_pred = model(X)
    l = loss(Y, y_pred)
    l.backward()

    with torch.no_grad():
        optimizer.step()

    optimizer.zero_grad()  # zero the parameter gradients

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}:w = {w[0][0]:.3f}, loss={l:.8f}")

print(f"Prediction After training: f(5)={model(X_test).item():.3f}")
