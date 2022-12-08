import torch
import torch.nn as nn


def train(model, device, train_loader, optimizer):
  criterion = nn.CrossEntropyLoss()
  model.train()
  for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = torch.argmax(output, dim=1)
            correct += torch.sum(pred == target)
    return 1 - correct / len(test_loader.dataset)
