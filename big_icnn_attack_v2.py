import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import h5py
import sys
sys.path.append('..')

from kyber import extract_msg
from big_icnn_model import InterconnectedKyberCNN

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Using device : {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────
ATTACK_FILE   = '../datasets/kem_dec_unprotected_8_attack.h5'
MODEL_WEIGHTS = 'big_icnn_best.pt'
BATCH_SIZE    = 256

# ── Load POIs ─────────────────────────────────────────────────────────────────
pois                = np.load('cnn_optimal_pois.npy').tolist()
pois_np             = np.array(pois)
SAMPLES_PER_CHANNEL = len(pois[0])
print(f"[+] POI shape           : {pois_np.shape}")

# ── Load attack file ──────────────────────────────────────────────────────────
print(f"\n[*] Loading attack dataset...")
with h5py.File(ATTACK_FILE, 'r') as f:
    raw_traces = f['traces'][:]
    raw_inputs = f['inputs'][:]

n_traces          = len(raw_traces)
n_secrets         = len(raw_inputs)
traces_per_secret = n_traces // n_secrets
print(f"[+] Total traces        : {n_traces:,}")
print(f"[+] Unique secrets      : {n_secrets}")
print(f"[+] Traces per secret   : {traces_per_secret}")

# ── Compute true labels ───────────────────────────────────────────────────────
print(f"\n[*] Computing true labels...")
true_labels = np.zeros((n_secrets, 32, 8), dtype=np.int64)

for i in range(n_secrets):
    data = raw_inputs[i]
    sk   = bytes(data[:1632])
    c    = bytes(data[1632:2400])
    msg  = extract_msg(sk, c)
    for byte_i in range(32):
        for bit_j in range(8):
            true_labels[i, byte_i, bit_j] = (int(msg[byte_i]) >> bit_j) & 1

print(f"[+] Labels shape        : {true_labels.shape}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\n[*] Loading model weights: {MODEL_WEIGHTS}")
model = InterconnectedKyberCNN(samples_per_channel=SAMPLES_PER_CHANNEL).to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
model.eval()
print(f"[+] Parameters          : {sum(p.numel() for p in model.parameters()):,}")
print(f"[+] Model ready!")

# ── Helper: build attack tensor ───────────────────────────────────────────────
def build_attack_tensor(n_avg):
    X = torch.zeros(n_secrets, 32, SAMPLES_PER_CHANNEL, dtype=torch.float32)
    for i in range(n_secrets):
        start     = i * traces_per_secret
        traces    = raw_traces[start:start + n_avg].astype(float)
        avg_trace = traces.mean(axis=0)
        multi_ch  = avg_trace[pois_np]
        mean      = np.mean(multi_ch, axis=1, keepdims=True)
        std       = np.std(multi_ch,  axis=1, keepdims=True) + 1e-8
        X[i]      = torch.tensor((multi_ch - mean) / std, dtype=torch.float32)
    return X

# ── Run attack ────────────────────────────────────────────────────────────────
print(f"\n[*] Running attack across averaging levels...\n")
y_true  = torch.tensor(true_labels, dtype=torch.long)
results = {}

for n_avg in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    if n_avg > traces_per_secret:
        break

    X_atk  = build_attack_tensor(n_avg)
    loader = DataLoader(TensorDataset(X_atk, y_true), batch_size=BATCH_SIZE, shuffle=False)

    total_correct = 0
    total_bits    = 0
    byte_correct  = [0] * 32
    byte_total    = [0] * 32

    with torch.no_grad():
        for traces, labels in loader:
            traces  = traces.to(DEVICE)
            labels  = labels.to(DEVICE)
            outputs = model(traces)
            for byte_i in range(32):
                for bit_j in range(8):
                    logits = outputs[:, byte_i, bit_j, :]
                    target = labels[:, byte_i, bit_j]
                    preds  = torch.argmax(logits, dim=1)
                    c      = (preds == target).sum().item()
                    total_correct        += c
                    byte_correct[byte_i] += c
                    total_bits           += target.numel()
                    byte_total[byte_i]   += target.numel()

    acc       = total_correct / total_bits * 100
    byte_accs = [byte_correct[i] / byte_total[i] * 100 for i in range(32)]
    results[n_avg] = {'overall': acc, 'per_byte': byte_accs}
    print(f"  N={n_avg:3d} | Bit acc: {acc:.2f}% | vs random: {acc-50:+.2f}%")

# ── Find best ─────────────────────────────────────────────────────────────────
best_n   = max(results, key=lambda n: results[n]['overall'])
best_acc = results[best_n]['overall']

print(f"\n{'='*60}")
print(f"  BEST: N={best_n} → {best_acc:.2f}% overall bit accuracy")
print(f"{'='*60}")

print(f"\nPer-byte breakdown at N={best_n}:")
print(f"{'Byte':>6} | {'Bit Acc':>8} | {'vs 50%':>8}")
print("-" * 30)
for i, acc in enumerate(results[best_n]['per_byte']):
    marker = " ←" if acc > 59 else ""
    print(f"  {i:02d}   | {acc:>7.2f}% | {acc-50:>+7.2f}%{marker}")

print(f"\n{'='*60}")
print(f"  FULL SUMMARY")
print(f"{'='*60}")
print(f"{'N avg':>6} | {'Bit Acc':>8} | {'vs Random':>10}")
print("-" * 35)
for n_avg, r in results.items():
    print(f"{n_avg:>6} | {r['overall']:>7.2f}% | {r['overall']-50:>+9.2f}%")

print(f"\n[+] Random baseline   : 50.00%")
print(f"[+] LDA single-trace  : ~0.37% (byte-level, 256-class)")
print(f"[+] CNN single-trace  : {results[1]['overall']:.2f}% (bit-level, binary)")
print(f"[+] CNN best N        : {best_acc:.2f}% (N={best_n})")

np.save('icnn_attack_results.npy', results)
print(f"\n[+] Saved: 'icnn_attack_results.npy'")

try:
    from google.colab import files
    files.download('icnn_attack_results.npy')
    print("[+] Downloaded!")
except:
    pass
