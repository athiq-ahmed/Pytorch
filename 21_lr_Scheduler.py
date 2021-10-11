import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1
model = nn.Linear(10, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# lamda1 = lambda epoch: epoch / 10
# scheduler = lr_scheduler.LambdaLR(optimizer, lamda1)

lamda1 = lambda epoch: 0.95
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lamda1)

print(optimizer.state_dict())

for epoch in range(5):
    # loss.backward()
    optimizer.step()
    # validate..
    scheduler.step()
    print(optimizer.state_dict()["param_groups"][0]["lr"])
