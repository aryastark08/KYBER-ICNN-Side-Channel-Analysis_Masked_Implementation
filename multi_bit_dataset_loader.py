import sys
sys.path.append('..')

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from kyber import extract_msg


class FullKyberTraceDataset(Dataset):
    """
    Binary formulation: predicts all 8 bits of each of 32 bytes.

    trace:  (32, 534)  float32
    labels: (32, 8)    int64    each value 0 or 1
    """

    def __init__(self, h5_path, pois, trace_limit=None):
        self.file   = h5py.File(h5_path, 'r',
                                rdcc_nbytes=512 * 1024 * 1024,
                                rdcc_nslots=521287)
        self.traces = self.file['traces']
        self.inputs = self.file['inputs']
        self.pois   = np.array(pois)
        self.length = len(self.traces)

        if trace_limit is not None:
            self.length = min(self.length, trace_limit)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # ── Trace ─────────────────────────────────────────────────────────────
        raw_trace = np.array(self.traces[idx])       # (6000,)
        multi_ch  = raw_trace[self.pois]             # (32, 534)

        mean = np.mean(multi_ch, axis=1, keepdims=True)
        std  = np.std(multi_ch,  axis=1, keepdims=True) + 1e-8
        norm = (multi_ch - mean) / std

        trace_tensor = torch.tensor(norm, dtype=torch.float32)  # (32, 534)

        # ── Labels ────────────────────────────────────────────────────────────
        data = self.inputs[idx]
        sk   = bytes(data[:1632])
        c    = bytes(data[1632:2400])
        msg  = extract_msg(sk, c)   # (32,) values 0-255

        # All 8 bits for each of 32 bytes → (32, 8)
        labels = torch.tensor(
            [[(int(msg[i]) >> bit) & 1 for bit in range(8)]
             for i in range(32)],
            dtype=torch.long
        )  # (32, 8)

        return trace_tensor, labels

    def close(self):
        self.file.close()
