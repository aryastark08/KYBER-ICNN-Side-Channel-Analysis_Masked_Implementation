import torch
import torch.nn as nn


class InterconnectedKyberCNN(nn.Module):
    """
    Lightweight Interconnected Kyber CNN.

    Reduced from 68M → ~2M parameters for practical CPU training.

    Input:  (Batch, 32, 534)
    Output: (Batch, 32, 8, 2)
    """

    def __init__(self, samples_per_channel=534):
        super().__init__()

        # Compute flat size after conv layers
        self._flat_size = self._compute_flat_size(samples_per_channel)

        # 32 shared-weight feature extractors (one per byte)
        # Using smaller filters and fewer channels
        self.sub_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=11, padding=5),  # 8 filters (was 16)
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(4),                              # bigger pool (was 2)

                nn.Conv1d(8, 16, kernel_size=11, padding=5), # 16 filters (was 32)
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(4),                              # bigger pool (was 2)

                nn.Flatten()
            ) for _ in range(32)
        ])

        # Single shared classifier head predicting all 8 bits at once
        # Shape: flat_size → 64 → 16 (8 bits × 2 classes)
        self.sub_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self._flat_size, 64),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(64, 16)  # 8 bits × 2 classes = 16 outputs
            ) for _ in range(32)
        ])

    def _compute_flat_size(self, samples):
        dummy = torch.zeros(1, 1, samples)
        out = nn.Sequential(
            nn.Conv1d(1, 8,  kernel_size=11, padding=5),
            nn.MaxPool1d(4),
            nn.Conv1d(8, 16, kernel_size=11, padding=5),
            nn.MaxPool1d(4),
            nn.Flatten()
        )(dummy)
        return out.shape[1]

    def forward(self, x):
        """
        Args:
            x: (Batch, 32, 534)
        Returns:
            out: (Batch, 32, 8, 2)
        """
        byte_outputs = []

        for i in range(32):
            channel_data = x[:, i, :].unsqueeze(1)        # (Batch, 1, 534)
            features     = self.sub_features[i](channel_data)  # (Batch, flat_size)
            logits_flat  = self.sub_classifiers[i](features)   # (Batch, 16)

            # Reshape to (Batch, 8, 2)
            logits = logits_flat.view(-1, 8, 2)
            byte_outputs.append(logits)

        # Stack → (Batch, 32, 8, 2)
        return torch.stack(byte_outputs, dim=1)
