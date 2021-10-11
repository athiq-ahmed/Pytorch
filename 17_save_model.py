import torch
import torch.nn as nn

# Important commands
"""
torch.save(arg, PATH)
torch.load(PATH)
model.load_state_dict(args)

"""

# Method 1
torch.save(model, PATH)
model = torch.load(PATH)
model.eval()

# Method 2
torch.save(model.state_dict(), PATH)
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()


class Model(nn.model):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model(n_input_features=6)
print(model.state_dict())

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())

# Method 1
FILE = "model.pth"
torch.save(model, FILE)
model = torch.load(FILE)
model.eval()

for param in model.parameters():
    print(param)


# Method 2
FILE = "model.pth"
torch.save(model.state_dict(), FILE)
model = Model(n_input_features=6)
model = model.load_state_dict(torch.load(FILE))
model.eval()

for param in model.parameters():
    print(param)


# Checkpoint
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
}

torch.save(checkpoint, "checkpoint.pth")
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
print(optimizer.state_dict())


# GPU
# Save on gpu and load on cpu
device = torch.device("cuda")
model.to(device)
torch.save(model.sate_dict(), PATH)

device = torch.device("cpu")
model = model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))


# Save on gpu and load on gpu
device = torch.device("cuda")
model.to(device)
torch.save(model.sate_dict(), PATH)

model = model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
model.to(device)


# Save on cpu and load on gpu
torch.save(model.sate_dict(), PATH)

device = torch.device("cuda")
model = model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
model.to(device)
