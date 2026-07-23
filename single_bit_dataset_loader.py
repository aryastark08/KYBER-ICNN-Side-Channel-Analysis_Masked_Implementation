import sys
sys.path.append('..')

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from kyber import extract_msg


class KyberTraceDataset(Dataset):
    """
    Single-bit recovery dataset.

    Returns per sample:
        trace: (1, n_pois)  float32  → single POI channel
        label: scalar       int64    → bit value (0 or 1)

    Args:
        h5_path    : path to HDF5 file
        pois       : list of 32 POI arrays (one per coefficient)
        trace_limit: max number of traces to use
        bit_index  : which byte AND which bit to recover (0-7)
    """

    def __init__(self, h5_path, pois, trace_limit=None, bit_index=0):
        print("A. Entering Dataset Init")
        self.file   = h5py.File(h5_path, 'r')
        print("B. HDF5 Opened")
        self.traces    = self.file['traces']
        print("C. Traces loaded")
        self.inputs    = self.file['inputs']
        print("D. Inputs loaded")

        self.bit_index  = bit_index

        # POIs for this specific coefficient/byte
        self.coeff_pois = pois[bit_index]
        print(f"E. Using POIs for byte {bit_index}: {len(self.coeff_pois)} points")

        self.length = len(self.traces)
        if trace_limit is not None:
            self.length = min(self.length, trace_limit)

        print(f"F. Dataset size: {self.length} traces")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Extract POI region from raw trace
        raw_trace = np.array(self.traces[idx])
        poi_trace = raw_trace[self.coeff_pois]   # (n_pois,)

        # 2. Z-score normalization
        poi_trace = (poi_trace - np.mean(poi_trace)) / (np.std(poi_trace) + 1e-8)

        # 3. Format for Conv1D: (1, n_pois)
        trace_tensor = torch.tensor(poi_trace, dtype=torch.float32).unsqueeze(0)

        # 4. Compute label
        data = self.inputs[idx]
        sk   = bytes(data[:1632])
        c    = bytes(data[1632:2400])
        msg  = extract_msg(sk, c)

        target_byte = int(msg[self.bit_index])

        # ── FIXED: use bit_index not hardcoded 0 ──────────────────────────────
        bit   = (target_byte >> self.bit_index) & 1
        label = torch.tensor(bit, dtype=torch.long)

        return trace_tensor, label

    def close(self):
        self.file.close()
