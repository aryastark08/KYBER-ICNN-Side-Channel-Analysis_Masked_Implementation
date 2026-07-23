import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
import sys
sys.path.append('..')
from kyber import extract_msg
from dataset_loader import KyberTraceDataset
from cnn_model import SimpleCNN

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
TRACE_LIMIT = 40000
BATCH_SIZE  = 128
EPOCHS      = 25
BITS        = list(range(8))   # 0 to 7

print(f"[*] Using device : {DEVICE}")
print(f"[*] Will train and attack bits: {BITS}")

# ── Load POIs ─────────────────────────────────────────────────────────────────
if os.path.exists('cnn_optimal_pois.npy'):
    pois = np.load('cnn_optimal_pois.npy').tolist()
    print(f"[+] CNN-optimized POIs: {np.array(pois).shape}")
else:
    pois = [
        [x + 67 * i - (i + 1) // 3 for x in range(233, 634)] +
        [x + 67 * i - (i + 1) // 3 for x in range(3367, 3500)]
        for i in range(32)
    ]
    print(f"[!] Using LDA POIs")

# ── Pre-load attack dataset ONCE (reused for all bits) ───────────────────────
print(f"\n[*] Pre-loading attack dataset...")
ATTACK_FILE = '../datasets/kem_dec_unprotected_8_attack.h5'

with h5py.File(ATTACK_FILE, 'r') as f:
    atk_traces = f['traces'][:]   # (51200, 6000)
    atk_inputs = f['inputs'][:]   # (200, 2432)

n_atk_traces      = len(atk_traces)
n_secrets         = len(atk_inputs)
traces_per_secret = n_atk_traces // n_secrets
print(f"[+] Attack: {n_secrets} secrets × {traces_per_secret} traces")

# ── Results storage ───────────────────────────────────────────────────────────
results = {}   # bit_index → {train_acc, attack_acc_n1, attack_acc_n8}

# ── Helper: train one bit ─────────────────────────────────────────────────────
def train_bit(bit_index):
    print(f"\n{'='*60}")
    print(f"  TRAINING BIT {bit_index}")
    print(f"{'='*60}")

    # Dataset
    dataset = KyberTraceDataset(
        '../datasets/kem_dec_unprotected_8.h5',
        pois=pois,
        trace_limit=TRACE_LIMIT,
        bit_index=bit_index
    )

    # Pre-allocate
    sample_trace, _ = dataset[0]
    n_pois          = sample_trace.shape[-1]
    X_tensor        = torch.zeros(TRACE_LIMIT, 1, n_pois, dtype=torch.float32)
    y_tensor        = torch.zeros(TRACE_LIMIT, dtype=torch.long)

    for i in range(len(dataset)):
        trace, label  = dataset[i]
        X_tensor[i]   = trace
        y_tensor[i]   = label

    print(f"[+] Labels: 0={(y_tensor==0).sum()} | 1={(y_tensor==1).sum()}")

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE == 'cuda')
    )

    # Model
    model     = SimpleCNN(n_pois=n_pois).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_acc   = 0.0
    no_improve = 0
    weights_path = f'weights_bit{bit_index}.pt'

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total   = 0

        for traces, labels in loader:
            traces = traces.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs  = model(traces)
            loss     = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds    = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        acc      = correct / total
        improved = "↑" if acc > best_acc else " "

        if acc > best_acc:
            best_acc   = acc
            no_improve = 0
            torch.save(model.state_dict(), weights_path)
        else:
            no_improve += 1

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Acc: {acc*100:.2f}% {improved} | "
              f"Best: {best_acc*100:.2f}%")

        if no_improve >= 8:
            print(f"  [!] Early stopping at epoch {epoch+1}")
            break

    print(f"[+] Bit {bit_index} training done. Best: {best_acc*100:.2f}%")
    return best_acc, weights_path, n_pois


# ── Helper: attack one bit ────────────────────────────────────────────────────
def attack_bit(bit_index, weights_path, n_pois):
    print(f"\n[*] Attacking bit {bit_index}...")

    coeff_pois = np.array(pois[bit_index])

    # Compute true labels for attack secrets
    true_labels = []
    for i in range(n_secrets):
        data = atk_inputs[i]
        sk   = bytes(data[:1632])
        c    = bytes(data[1632:2400])
        msg  = extract_msg(sk, c)
        bit  = (int(msg[bit_index]) >> bit_index) & 1
        true_labels.append(bit)
    true_labels = np.array(true_labels)

    # Load model
    model = SimpleCNN(n_pois=n_pois).to(DEVICE)
    model.load_state_dict(torch.load(
        weights_path, map_location=DEVICE, weights_only=True
    ))
    model.eval()

    # Attack with different averaging
    attack_results = {}

    for n_avg in [1, 8, 256]:
        if n_avg > traces_per_secret:
            continue
        correct = 0
        with torch.no_grad():
            for i in range(n_secrets):
                start     = i * traces_per_secret
                traces    = atk_traces[start:start + n_avg].astype(float)
                avg_trace = traces.mean(axis=0)
                poi       = avg_trace[coeff_pois]
                poi       = (poi - poi.mean()) / (poi.std() + 1e-8)
                x         = torch.tensor(poi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                out       = model(x)
                pred      = torch.argmax(out, dim=1).item()
                if pred == true_labels[i]:
                    correct += 1

        acc = correct / n_secrets
        attack_results[n_avg] = acc
        print(f"  N={n_avg:3d} avg → {acc*100:.2f}% ({(acc-0.5)*100:+.2f}% vs random)")

    return attack_results


# ── Main loop: train and attack all bits ──────────────────────────────────────
all_results = {}

for bit in BITS:
    train_acc, weights_path, n_pois = train_bit(bit)
    attack_accs                      = attack_bit(bit, weights_path, n_pois)

    all_results[bit] = {
        'train_acc'  : train_acc,
        'attack_n1'  : attack_accs.get(1,   0),
        'attack_n8'  : attack_accs.get(8,   0),
        'attack_n256': attack_accs.get(256, 0),
    }

# ── Final comparison table ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("          FULL BIT COMPARISON TABLE")
print("=" * 70)
print(f"{'Bit':>4} | {'Train':>8} | {'Atk N=1':>9} | {'Atk N=8':>9} | {'Atk N=256':>10}")
print("-" * 70)

best_bit_n1   = max(all_results, key=lambda b: all_results[b]['attack_n1'])
best_bit_n8   = max(all_results, key=lambda b: all_results[b]['attack_n8'])
best_bit_n256 = max(all_results, key=lambda b: all_results[b]['attack_n256'])

for bit, r in all_results.items():
    marker_n1   = " ←best" if bit == best_bit_n1   else ""
    marker_n8   = " ←best" if bit == best_bit_n8   else ""
    marker_n256 = " ←best" if bit == best_bit_n256 else ""
    print(f"  {bit:2d} | "
          f"{r['train_acc']*100:>7.2f}% | "
          f"{r['attack_n1']*100:>8.2f}%{marker_n1} | "
          f"{r['attack_n8']*100:>8.2f}%{marker_n8} | "
          f"{r['attack_n256']*100:>9.2f}%{marker_n256}")

print("-" * 70)
print(f"{'Mean':>4} | "
      f"{np.mean([r['train_acc']   for r in all_results.values()])*100:>7.2f}% | "
      f"{np.mean([r['attack_n1']   for r in all_results.values()])*100:>8.2f}% | "
      f"{np.mean([r['attack_n8']   for r in all_results.values()])*100:>8.2f}% | "
      f"{np.mean([r['attack_n256'] for r in all_results.values()])*100:>9.2f}%")
print("=" * 70)

print(f"\n[+] Best single-trace bit : {best_bit_n1} ({all_results[best_bit_n1]['attack_n1']*100:.2f}%)")
print(f"[+] Best 8-avg bit        : {best_bit_n8} ({all_results[best_bit_n8]['attack_n8']*100:.2f}%)")
print(f"[+] Random baseline       : 50.00%")

# Save results
np.save('all_bits_results.npy', all_results)
print(f"\n[+] Results saved: 'all_bits_results.npy'")
