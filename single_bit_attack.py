import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py
import sys
sys.path.append('..')
from kyber import extract_msg
from cnn_model import SimpleCNN

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = 'cpu'
BIT_INDEX   = 2
N_AVERAGE   = 8

# ── Load POIs ─────────────────────────────────────────────────────────────────
import os
if os.path.exists('cnn_optimal_pois.npy'):
    pois = np.load('cnn_optimal_pois.npy').tolist()
    print(f"[+] Using CNN-optimized POIs: {np.array(pois).shape}")
else:
    pois = [
        [x + 67 * i - (i + 1) // 3 for x in range(233, 634)] +
        [x + 67 * i - (i + 1) // 3 for x in range(3367, 3500)]
        for i in range(32)
    ]
    print(f"[!] Using LDA POIs")

coeff_pois = np.array(pois[BIT_INDEX])
n_pois     = len(coeff_pois)

print(f"[*] Attacking bit index : {BIT_INDEX}")
print(f"[*] POIs per channel    : {n_pois}")

# ── Load attack dataset ───────────────────────────────────────────────────────
print(f"\n[*] Loading attack dataset...")
ATTACK_FILE = '../datasets/kem_dec_unprotected_8_attack.h5'

with h5py.File(ATTACK_FILE, 'r') as f:
    all_traces = f['traces'][:]
    all_inputs = f['inputs'][:]

n_traces          = len(all_traces)
n_inputs          = len(all_inputs)
traces_per_secret = n_traces // n_inputs

print(f"[+] Total traces        : {n_traces}")
print(f"[+] Unique secrets      : {n_inputs}")
print(f"[+] Traces per secret   : {traces_per_secret}")

# ── Compute true labels ───────────────────────────────────────────────────────
print(f"\n[*] Computing true labels...")
true_labels = []

for i in range(n_inputs):
    data = all_inputs[i]
    sk   = bytes(data[:1632])
    c    = bytes(data[1632:2400])
    msg  = extract_msg(sk, c)
    bit  = (int(msg[BIT_INDEX]) >> BIT_INDEX) & 1
    true_labels.append(bit)

true_labels = np.array(true_labels)
print(f"[+] Labels: 0={(true_labels==0).sum()} | 1={(true_labels==1).sum()}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\n[*] Loading model weights...")
model = SimpleCNN(n_pois=n_pois).to(DEVICE)
model.load_state_dict(torch.load(
    'single_bit_prototype_weights.pt',
    map_location=DEVICE,
    weights_only=True
))
model.eval()
print(f"[+] Model loaded!")

# ── Attack: Varying average counts ────────────────────────────────────────────
print(f"\n[*] Testing different trace averaging amounts...")
print(f"{'Avg N':>8} | {'Accuracy':>10} | {'vs Random':>10}")
print("-" * 35)

for n_avg in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    if n_avg > traces_per_secret:
        break
    correct = 0
    with torch.no_grad():
        for i in range(n_inputs):
            start     = i * traces_per_secret
            traces    = all_traces[start:start + n_avg].astype(float)
            avg_trace = traces.mean(axis=0)
            poi       = avg_trace[coeff_pois]
            poi       = (poi - poi.mean()) / (poi.std() + 1e-8)
            x         = torch.tensor(poi, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            out       = model(x)
            pred      = torch.argmax(out, dim=1).item()
            if pred == true_labels[i]:
                correct += 1

    acc = correct / n_inputs
    print(f"{n_avg:>8} | {acc*100:>9.2f}% | {(acc-0.5)*100:>+9.2f}%")

print("=" * 50)
