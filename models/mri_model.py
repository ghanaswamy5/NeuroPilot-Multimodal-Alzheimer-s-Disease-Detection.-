import torch
import torch.nn as nn

class MRI_Model(nn.Module):
    def __init__(self):
        super(MRI_Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),   # (224 → 222)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (222 → 111)

            nn.Conv2d(16, 32, kernel_size=3),  # (111 → 109)
            nn.ReLU(),
            nn.MaxPool2d(2)                    # (109 → 54)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 54 * 54, 128),
            nn.ReLU(),
            nn.Linear(128, 3)   # 🔥 3 classes: AD, MCI, CN
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x