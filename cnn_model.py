import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Single-bit recovery CNN.

    Input:  (Batch, 1, n_pois)   - single channel POI trace
    Output: (Batch, 2)           - binary classification (bit=0 or bit=1)
    """

    def __init__(self, n_pois=534):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten()
        )

        # Dynamically compute flat size instead of hardcoding
        self._flat_size = self._compute_flat_size(n_pois)

        self.classifier = nn.Sequential(
            nn.Linear(self._flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def _compute_flat_size(self, n_pois):
        dummy = torch.zeros(1, 1, n_pois)
        out   = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5),
            nn.MaxPool1d(2),
            nn.Flatten()
        )(dummy)
        return out.shape[1]

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
