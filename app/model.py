import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        return h

    def forward(self, x):
        h = self.forward_features(x)
        return self.fc2(h), h


def load_model(path: str, device: torch.device) -> MNISTCNN:
    model = MNISTCNN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
