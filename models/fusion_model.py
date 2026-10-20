import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        # EEG + MRI features combined
        self.fc = nn.Sequential(
            nn.Linear(6, 16),   # 3 EEG probs + 3 MRI probs
            nn.ReLU(),
            nn.Linear(16, 3)    # AD, CN, MCI
        )

    def forward(self, eeg_probs, mri_probs):
        x = torch.cat((eeg_probs, mri_probs), dim=1)
        return self.fc(x)