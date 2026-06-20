import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataset_loader import KyberTraceDataset
from cnn_model import SimpleCNN

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Using device: {DEVICE}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
TRACE_LIMIT = 40000
BATCH_SIZE  = 128
EPOCHS      = 25
BIT_INDEX   = 2   # which bit to recover (0-7)

# ── POIs ──────────────────────────────────────────────────────────────────────
import os
if os.path.exists('cnn_optimal_pois.npy'):
    # Use CNN-optimized POIs (better signal for CNN)
    pois = np.load('cnn_optimal_pois.npy').tolist()
    print(f"[+] Using CNN-optimized POIs: {np.array(pois).shape}")
else:
    # Fallback to LDA POIs
    pois = [
        [x + 67 * i - (i + 1) // 3 for x in range(233, 634)] +
        [x + 67 * i - (i + 1) // 3 for x in range(3367, 3500)]
        for i in range(32)
    ]
    print(f"[!] cnn_optimal_pois.npy not found, using LDA POIs")

print(f"[*] Recovering bit index : {BIT_INDEX}")
print(f"[*] Trace limit          : {TRACE_LIMIT}")

# ── Dataset ───────────────────────────────────────────────────────────────────
print("\n[*] Creating dataset...")
raw_file_dataset = KyberTraceDataset(
    '../datasets/kem_dec_unprotected_8.h5',
    pois=pois,
    trace_limit=TRACE_LIMIT,
    bit_index=BIT_INDEX
)

# ── RAM Pre-load (memory efficient) ───────────────────────────────────────────
print(f"[*] Pre-loading {TRACE_LIMIT} traces into RAM...")

# Pre-allocate directly to avoid peak RAM spike
sample_trace, sample_label = raw_file_dataset[0]
X_tensor = torch.zeros(TRACE_LIMIT, *sample_trace.shape, dtype=torch.float32)
y_tensor = torch.zeros(TRACE_LIMIT, dtype=torch.long)

for i in range(len(raw_file_dataset)):
    if i % 5000 == 0 and i > 0:
        print(f"    -> Loaded {i}/{TRACE_LIMIT} traces...")
    trace, label  = raw_file_dataset[i]
    X_tensor[i]   = trace
    y_tensor[i]   = label

print(f"[+] X_tensor : {X_tensor.shape}")
print(f"[+] y_tensor : {y_tensor.shape}")
print(f"[+] Label distribution: 0={( y_tensor==0).sum()} | 1={(y_tensor==1).sum()}")

# ── DataLoader ────────────────────────────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(X_tensor, y_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(DEVICE == 'cuda')
)

# ── Model ─────────────────────────────────────────────────────────────────────
print("\n[*] Initializing SimpleCNN...")
model     = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

total_params = sum(p.numel() for p in model.parameters())
print(f"[+] Parameters: {total_params:,}")

# ── Training ──────────────────────────────────────────────────────────────────
print(f"\n[*] Training for {EPOCHS} epochs (recovering bit {BIT_INDEX})...\n")

best_acc   = 0.0
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for traces, labels in train_loader:
        traces = traces.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs     = model(traces)
        loss        = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        predictions  = torch.argmax(outputs, dim=1)
        correct     += (predictions == labels).sum().item()
        total       += labels.size(0)

    accuracy  = correct / total
    improved  = "↑" if accuracy > best_acc else " "

    if accuracy > best_acc:
        best_acc   = accuracy
        no_improve = 0
        torch.save(model.state_dict(), 'single_bit_prototype_weights.pt')
    else:
        no_improve += 1

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Loss: {total_loss:.4f} | "
          f"Acc: {accuracy:.4f} ({accuracy*100:.2f}%) {improved} | "
          f"Best: {best_acc:.4f}")

    if no_improve >= 8:
        print(f"\n[!] Early stopping after {epoch+1} epochs.")
        break

print(f"\n[+] Best accuracy    : {best_acc*100:.2f}%")
print(f"[+] Random baseline  : 50.00%")
print(f"[+] Improvement      : +{(best_acc-0.5)*100:.2f}%")
print(f"[+] Weights saved    : 'single_bit_prototype_weights.pt'")
