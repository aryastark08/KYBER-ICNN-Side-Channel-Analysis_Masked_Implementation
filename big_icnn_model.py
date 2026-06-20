import torch
import torch.nn as nn


class InterconnectedKyberCNN(nn.Module):
    """
    RAM-Safe Binary CNN - fits comfortably in 16GB RAM.

    Target: ~5-10M parameters
    Input:  (Batch, 32, 534)
    Output: (Batch, 32, 8, 2)
    """

    def __init__(self, samples_per_channel=534):
        super().__init__()

        self._flat_size = self._compute_flat_size(samples_per_channel)

        # 32 feature extractors - moderate size
        self.sub_features = nn.ModuleList([
            nn.Sequential(
                # Block 1
                nn.Conv1d(1, 16, kernel_size=11, padding=5),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.AvgPool1d(2),

                # Block 2
                nn.Conv1d(16, 32, kernel_size=11, padding=5),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(2),

                # Block 3
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AvgPool1d(2),

                nn.Flatten()
            ) for _ in range(32)
        ])

        # 32 classifiers - moderate hidden size
        self.sub_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self._flat_size, 128),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(128, 16)   # 8 bits × 2 classes
            ) for _ in range(32)
        ])

    def _compute_flat_size(self, samples):
        dummy = torch.zeros(1, 1, samples)
        out = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, kernel_size=11, padding=5),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.AvgPool1d(2),
            nn.Flatten()
        )(dummy)
        return out.shape[1]

    def forward(self, x):
        byte_outputs = []
        for i in range(32):
            ch       = x[:, i, :].unsqueeze(1)
            features = self.sub_features[i](ch)
            logits   = self.sub_classifiers[i](features).view(-1, 8, 2)
            byte_outputs.append(logits)
        return torch.stack(byte_outputs, dim=1)  # (Batch, 32, 8, 2)