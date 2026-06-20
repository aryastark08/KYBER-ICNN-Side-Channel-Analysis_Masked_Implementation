import numpy as np
import h5py
import sys
sys.path.append('..')
from kyber import extract_msg

print("[*] Finding strongest signal regions...")

with h5py.File('../datasets/kem_dec_unprotected_8.h5', 'r') as f:
    traces = f['traces'][:2000]    # first 2000 traces
    inputs = f['inputs'][:2000]

    # Compute labels for first 2000
    labels = []
    for i in range(2000):
        data = inputs[i]
        sk   = bytes(data[:1632])
        c    = bytes(data[1632:2400])
        msg  = extract_msg(sk, c)
        labels.append(int(msg[0]))  # byte 0 label

    labels = np.array(labels)

    # Compute correlation between each time point and byte 0 label
    correlations = np.zeros(6000)
    for t in range(6000):
        correlations[t] = abs(np.corrcoef(
            traces[:, t].astype(float),
            labels.astype(float)
        )[0, 1])

    # Find top 50 most correlated time points
    top_50 = np.argsort(correlations)[-50:][::-1]

    print(f"\nTop 10 most correlated time points (byte 0):")
    for rank, idx in enumerate(top_50[:10]):
        print(f"  Rank {rank+1:2d}: time={idx:4d} | correlation={correlations[idx]:.6f}")

    print(f"\nCurrent POI range 1: 233-633")
    print(f"Current POI range 2: 3367-3499")

    # Check correlation in current POI regions
    poi_corr_1 = correlations[233:634].mean()
    poi_corr_2 = correlations[3367:3500].mean()
    overall    = correlations.mean()
    np.save('correlation_profile.npy', correlations)
    print('[+] Saved: correlation_profile.npy')
    print(f"\nCorrelation analysis:")
    print(f"  Overall mean    : {overall:.6f}")
    print(f"  POI region 1    : {poi_corr_1:.6f}")
    print(f"  POI region 2    : {poi_corr_2:.6f}")
    print(f"  Best time point : {correlations.max():.6f}")