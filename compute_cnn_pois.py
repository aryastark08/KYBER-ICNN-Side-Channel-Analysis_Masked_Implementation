import numpy as np
import h5py
import sys
sys.path.append('..')
from kyber import extract_msg

print("[*] Computing optimal CNN POIs for all 32 bytes...")

N_TRACES   = 5000   # use 5000 traces for POI search
TOP_N      = 534    # same number of POIs as before

with h5py.File('../datasets/kem_dec_unprotected_8.h5', 'r') as f:
    traces = f['traces'][:N_TRACES].astype(float)
    inputs = f['inputs'][:N_TRACES]

    # Compute labels for all 32 bytes
    print("[*] Computing labels...")
    all_labels = np.zeros((N_TRACES, 32), dtype=np.int32)
    for i in range(N_TRACES):
        if i % 1000 == 0:
            print(f"    -> {i}/{N_TRACES}")
        data = inputs[i]
        sk   = bytes(data[:1632])
        c    = bytes(data[1632:2400])
        msg  = extract_msg(sk, c)
        all_labels[i] = [int(msg[j]) for j in range(32)]

    print("[*] Computing correlations for each byte...")
    cnn_pois = []

    for byte_i in range(32):
        labels = all_labels[:, byte_i]

        # Compute correlation at each time point
        correlations = np.zeros(6000)
        for t in range(6000):
            correlations[t] = abs(np.corrcoef(
                traces[:, t],
                labels.astype(float)
            )[0, 1])

        # Take top N most correlated points
        top_indices = np.argsort(correlations)[-TOP_N:]
        top_indices = np.sort(top_indices)   # keep time order
        cnn_pois.append(top_indices.tolist())

        print(f"  Byte {byte_i:02d}: "
              f"best corr={correlations.max():.4f} | "
              f"mean corr={correlations[top_indices].mean():.4f}")

    # Save the new POIs
    np.save('cnn_optimal_pois.npy', np.array(cnn_pois))
    print("\n[+] Saved: cnn_optimal_pois.npy")
    print(f"[+] POI shape: {np.array(cnn_pois).shape}")  # (32, 534)