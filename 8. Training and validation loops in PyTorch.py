import torch

def train_one_step(model, data, optimizer):
    optimizer.zero_grad()
    for k, v in data.items():  # if data is not in same format of model
        data["k"] = v.to("cuda")
    loss = model(x=data["x"], y=data["y"])
    # loss = model(**data)    # if data is in same format of model
    loss.backward()
    optimizer.step()
    return loss


def train_one_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch_index, data in enumerate(data_loader):
        loss = train_one_step(model, data, optimizer)
        scheduler.step()
        total_loss += loss
    return total_loss


def validate_one_step(model, data):
    for k, v in data.items():  # if data is not in same format of model
        data["k"] = v.to("cuda")
    loss = model(**data)    # if data is in same format of model
    return loss


def validate_one_epoch(model, data_loader):
    model.eval()
    total_loss = 0
    for batch_index, data in enumerate(data_loader):
        with torch.no_grad():
            loss = validate_one_step(model, data)
        total_loss += loss
    return total_loss
