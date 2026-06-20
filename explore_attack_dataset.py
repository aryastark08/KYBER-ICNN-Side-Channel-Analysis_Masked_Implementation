import h5py
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('..')

file_path = r"E:\Radbound University - Internship\Research-KYBER\datasets\kem_dec_unprotected_8_attack.h5"

# ── Safety check ───────────────────────────────────────────────────────────────
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    print("Update file_path at the top of this script")
    exit()

print(f"File found! Size: {os.path.getsize(file_path) / (1024**3):.2f} GB\n")

with h5py.File(file_path, "r") as f:

    # ── Structure ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("TOP LEVEL KEYS")
    print("=" * 70)
    print(list(f.keys()))

    print("\n" + "=" * 70)
    print("FULL STRUCTURE")
    print("=" * 70)

    def explore(group, level=0):
        indent = "  " * level
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                print(f"{indent}[Dataset] {key}")
                print(f"{indent}  Shape : {item.shape}")
                print(f"{indent}  Dtype : {item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"{indent}[Group] {key}/")
                explore(item, level + 1)

    explore(f)

    # ── Compare with training dataset ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON: TRAINING vs ATTACK DATASET")
    print("=" * 70)

    print(f"""
  TRAINING (kem_dec_unprotected_8.h5):
    inputs : (100000, 2432)   uint8
    traces : (100000, 6000)   int8

  ATTACK   (kem_dec_unprotected_8_attack.h5):""")

    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Dataset):
            print(f"    {key:8s}: {str(item.shape):20s}  {item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"    {key}/ (group):")
            for subkey in item.keys():
                sub = item[subkey]
                print(f"      {subkey:8s}: {str(sub.shape):20s}  {sub.dtype}")

    # ── Load inputs and traces ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATA PREVIEW")
    print("=" * 70)

    # Handle both flat and grouped structure
    if 'inputs' in f:
        inputs = f['inputs']
        traces = f['traces']
    elif 'fixed' in f:
        inputs = f['fixed']['inputs']
        traces = f['fixed']['traces']
    else:
        # Try first available key
        keys = list(f.keys())
        print(f"Unexpected structure, keys: {keys}")
        exit()

    print(f"\nINPUTS (first 5 rows, first 12 cols):")
    df_inputs = pd.DataFrame(
        inputs[:5, :12],
        columns=[f"col_{i}" for i in range(12)]
    )
    df_inputs.index.name = "sample"
    print(df_inputs.to_string())

    print(f"\nTRACES (first 5 rows, first 20 time samples):")
    df_traces = pd.DataFrame(
        traces[:5, :20],
        columns=[f"t{i}" for i in range(20)]
    )
    df_traces.index.name = "trace"
    print(df_traces.to_string())

    # ── Stats ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    inp_arr = inputs[:1000]
    trc_arr = traces[:1000]

    print(f"""
  Inputs (first 1000):
    Min  : {inp_arr.min()}
    Max  : {inp_arr.max()}
    Mean : {inp_arr.mean():.4f}

  Traces (first 1000):
    Min  : {trc_arr.min()}
    Max  : {trc_arr.max()}
    Mean : {trc_arr.mean():.4f}
    Std  : {trc_arr.std():.4f}
    """)

    # ── Key difference from training ───────────────────────────────────────────
    print("=" * 70)
    print("WHAT IS DIFFERENT IN ATTACK DATASET?")
    print("=" * 70)

    print(f"""
  The attack dataset is used to EVALUATE your trained model.

  TRAINING dataset:
    - Random inputs each trace
    - Model LEARNS from these
    - You know the labels (compute via extract_msg)

  ATTACK dataset:
    - May have DIFFERENT structure (e.g. averaged traces)
    - Used to TEST the model on new unseen data
    - The researcher's LDA used this for evaluation

  Key thing to check:
    - Is input structure the same? (same 2432 bytes?)
    - Is trace length the same?   (same 6000 samples?)
    - How many total samples?

  Total samples in attack dataset: {len(traces):,}
    """)

    # ── Check if structure matches for extract_msg ─────────────────────────────
    print("=" * 70)
    print("CHECKING EXTRACT_MSG COMPATIBILITY")
    print("=" * 70)

    try:
        from kyber import extract_msg

        data = inputs[0]
        print(f"Input row 0 length : {len(data)}")
        print(f"SK bytes [0:1632]  : {len(data[:1632])} bytes")
        print(f"C  bytes [1632:2400]: {len(data[1632:2400])} bytes")

        sk  = bytes(data[:1632])
        c   = bytes(data[1632:2400])
        msg = extract_msg(sk, c)

        print(f"\nextract_msg works on attack dataset!")
        print(f"  msg shape : {msg.shape}")
        print(f"  msg[:8]   : {list(msg[:8])}")

    except Exception as e:
        print(f"extract_msg check failed: {e}")
        print("Attack dataset may have different input format")
