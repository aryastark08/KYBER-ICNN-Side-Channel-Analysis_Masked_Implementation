import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from big_icnn_dataset_loader import FullKyberTraceDataset
from big_icnn_model import InterconnectedKyberCNN

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Using device: {DEVICE}")

if DEVICE == 'cuda':
    print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[*] GPU RAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    torch.set_num_threads(4)
    torch.backends.nnpack.enabled = False

# ── Hyperparameters ───────────────────────────────────────────────────────────
TRACE_LIMIT   = 90000   # increased from 60k
BATCH_SIZE    = 256
EPOCHS        = 50
LEARNING_RATE = 0.0005  # reduced from 0.001
COEFFS        = 32

if DEVICE == 'cpu':
    TRACE_LIMIT = 60000
    EPOCHS      = 15
    print(f"[*] CPU mode: {TRACE_LIMIT} traces, {EPOCHS} epochs")

# ── POIs ──────────────────────────────────────────────────────────────────────
pois = np.load('cnn_optimal_pois.npy').tolist()
SAMPLES_PER_CHANNEL = len(pois[0])

max_poi = max(max(p) for p in pois)
print(f"\n[+] CNN-optimized POIs  : {np.array(pois).shape}")
print(f"[*] Max POI index       : {max_poi}")
print(f"[*] Samples per channel : {SAMPLES_PER_CHANNEL}")
print(f"[*] Trace limit         : {TRACE_LIMIT}")
print(f"[*] Learning rate       : {LEARNING_RATE}")
print(f"[*] Epochs              : {EPOCHS}")

# ── RAM estimate ──────────────────────────────────────────────────────────────
x_ram = TRACE_LIMIT * 32 * SAMPLES_PER_CHANNEL * 4 / (1024**3)
y_ram = TRACE_LIMIT * 32 * 8 * 8 / (1024**3)
print(f"\n[*] RAM: X={x_ram:.2f}GB | y={y_ram:.2f}GB | Total={x_ram+y_ram:.2f}GB")

# ── Check model size ──────────────────────────────────────────────────────────
test_model   = InterconnectedKyberCNN(samples_per_channel=SAMPLES_PER_CHANNEL)
total_params = sum(p.numel() for p in test_model.parameters())
print(f"[*] Parameters: {total_params:,}")
del test_model

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"\n[*] Loading dataset...")
raw_dataset = FullKyberTraceDataset(
    '../datasets/kem_dec_unprotected_8.h5',
    pois=pois,
    trace_limit=TRACE_LIMIT
)

# ── RAM Pre-load (memory efficient) ───────────────────────────────────────────
print(f"[*] Pre-allocating tensors (no peak RAM spike)...")
sample_trace, sample_label = raw_dataset[0]
X_tensor = torch.zeros(TRACE_LIMIT, *sample_trace.shape, dtype=torch.float32)
y_tensor = torch.zeros(TRACE_LIMIT, *sample_label.shape, dtype=torch.long)

for i in range(TRACE_LIMIT):
    if i % 10000 == 0 and i > 0:
        print(f"    -> {i}/{TRACE_LIMIT}...")
    trace, labels  = raw_dataset[i]
    X_tensor[i]    = trace
    y_tensor[i]    = labels

print(f"[+] X_tensor : {X_tensor.shape}")
print(f"[+] y_tensor : {y_tensor.shape}")

# ── DataLoader ────────────────────────────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(X_tensor, y_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE == 'cuda')
)

# ── Model ─────────────────────────────────────────────────────────────────────
print("\n[*] Initializing model...")
model     = InterconnectedKyberCNN(samples_per_channel=SAMPLES_PER_CHANNEL).to(DEVICE)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)
criterion = nn.CrossEntropyLoss()
print(f"[+] Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Shape check
with torch.no_grad():
    dummy = X_tensor[:2].to(DEVICE)
    out   = model(dummy)
    assert out.shape == (2, 32, 8, 2), f"Wrong: {out.shape}"
    print(f"[+] Output shape: {out.shape} ✓")

# ── Training ──────────────────────────────────────────────────────────────────
print(f"\n[*] Training (LR={LEARNING_RATE}, traces={TRACE_LIMIT})...\n")

best_acc   = 0.0
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss    = 0.0
    total_correct = 0
    total_bits    = 0

    for step, (traces, labels) in enumerate(train_loader):
        traces = traces.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(traces)

        loss = 0
        for byte_i in range(32):
            for bit_j in range(8):
                logits = outputs[:, byte_i, bit_j, :]
                target = labels[:, byte_i, bit_j]
                loss  += criterion(logits, target)
                preds          = torch.argmax(logits, dim=1)
                total_correct += (preds == target).sum().item()
                total_bits    += target.numel()

        loss = loss / 256
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if DEVICE == 'cpu' and step % 50 == 0:
            time.sleep(0.01)

    train_acc = (total_correct / total_bits) * 100
    improved  = "↑" if train_acc > best_acc else " "

    if train_acc > best_acc:
        best_acc   = train_acc
        no_improve = 0
        torch.save(model.state_dict(), 'big_icnn_best.pt')
    else:
        no_improve += 1

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% {improved} | Best: {best_acc:.2f}%")

    if no_improve >= 10:
        print(f"\n[!] Early stopping at epoch {epoch+1}.")
        break

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[*] Loading best model and evaluating...")
model.load_state_dict(torch.load('big_icnn_best.pt', map_location=DEVICE))
model.eval()

byte_correct = [0] * 32
byte_total   = [0] * 32

with torch.no_grad():
    for traces, labels in train_loader:
        traces = traces.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(traces)
        for byte_i in range(32):
            for bit_j in range(8):
                logits = outputs[:, byte_i, bit_j, :]
                target = labels[:, byte_i, bit_j]
                preds  = torch.argmax(logits, dim=1)
                byte_correct[byte_i] += (preds == target).sum().item()
                byte_total[byte_i]   += target.numel()

byte_accuracies = [byte_correct[i] / byte_total[i] for i in range(32)]

print(f"\nPer-byte bit accuracy:")
print(f"{'Byte':>6} | {'Bit Acc':>8} | {'vs 50%':>8}")
print("-" * 32)
for i, acc in enumerate(byte_accuracies):
    gain   = (acc - 0.5) * 100
    marker = " ←" if acc > 0.59 else ""
    print(f"  {i:02d}   | {acc*100:>7.2f}% | {gain:>+7.2f}%{marker}")

overall = np.mean(byte_accuracies) * 100
print(f"\nOverall  : {overall:.2f}%")
print(f"Baseline : 50.00%")
print(f"Gain     : +{overall - 50:.2f}%")

np.save('big_icnn_byte_accuracies.npy', np.array(byte_accuracies))
print("\n[+] Saved: 'big_icnn_byte_accuracies.npy'")

try:
    from google.colab import files
    files.download('big_icnn_best.pt')
    files.download('big_icnn_byte_accuracies.npy')
    print("[+] Downloaded!")
except:
    pass
