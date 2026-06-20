# Deep Learning-Based Message Recovery Against ML-KEM

**Research Internship — Radboud University, Nijmegen**  
Supervised by Ms. Azade Rezaeezade and Prof. Lejla Batina

---

## What is this?

This repository contains the CNN-based side-channel attack code I developed during my research internship. The goal was to recover secret message bits from a single power trace of the ML-KEM (Kyber-512) hardware decoder — first on an unprotected implementation, and then on a masked one.

In short: we show that a neural network can extract useful information from one power measurement, where classical statistical attacks (LDA) completely fail at single-trace recovery.

**Key results at a glance:**

| Model | Single-trace accuracy | Traces needed |
|-------|----------------------|---------------|
| LDA (baseline, reproduced) | ~0.37% (below random) | 4–16 |
| Single-Bit CNN (this work) | 64% best bit | **1** |
| Multi-Bit CNN (this work) | 60.25% across 256 bits | **1** |

---

## What's in this repo

```
├── cnn_model.py                  # Single-Bit CNN architecture
├── big_icnn_model.py             # Multi-Bit CNN architecture (17.9M params)
├── dataset_loader.py             # Dataset loader for Single-Bit CNN
├── big_icnn_dataset_loader.py    # Dataset loader for Multi-Bit CNN
├── create_icnn_poi.py            # Compute CNN-optimized Points of Interest
├── new_poi.py                    # POI correlation analysis
├── check_poi.py                  # POI variance check
├── train_cnn.py                  # Train Single-Bit CNN (one bit at a time)
├── train_attack_all_bits.py      # Train + attack all 8 bits automatically
├── big_icnn_train_v2.py          # Train Multi-Bit CNN (final version)
├── attack_cnn_v2.py              # Attack script for Single-Bit CNN
├── big_icnn_attack_v2.py         # Attack script for Multi-Bit CNN
├── plot_single_bit_results.py    # Plot per-bit accuracy comparison
├── plot_poi_comparison.py        # Plot LDA vs CNN POI comparison
├── cnn_optimal_pois.npy          # Pre-computed CNN POIs (ready to use)
├── all_bits_results.npy          # Single-Bit CNN results
├── big_icnn_byte_accuracies.npy  # Multi-Bit CNN per-byte training accuracy
└── icnn_attack_results.npy       # Multi-Bit CNN attack results
```

The scripts depend on `kyber.py` and the dataset structure from the original attack framework by Dobias & Malina, included here as a git submodule.

---

## Before you start

You will need:
- Python 3.8+
- A Linux environment (scripts were developed and tested on Linux)
- A GPU is strongly recommended for training the Multi-Bit CNN (we used Google Colab with a Tesla T4)
- The datasets from Zenodo (not included here due to file size)

---

## Step 1 — Clone the repo

```bash
git clone --recurse-submodules https://github.com/aryastark08/KYBER-ICNN-Side-Channel-Analysis_Masked_Implementation.git
cd KYBER-ICNN-Side-Channel-Analysis_Masked_Implementation
```

The `--recurse-submodules` flag is important — it also pulls in the original attack framework that the scripts depend on.

If you forgot the flag:
```bash
git submodule update --init --recursive
```

---

## Step 2 — Set up the directory structure

The scripts expect to run from inside `message-recovery-attack-on-ml-kem/icnn_experiments/` and look for datasets one level up in `../datasets/`. Set it up like this:

```
message-recovery-attack-on-ml-kem/
├── datasets/               ← put your .h5 files here
│   ├── kem_dec_unprotected_8.h5
│   ├── kem_dec_unprotected_8_attack.h5
│   └── sec_decoder_masked.h5
├── icnn_experiments/       ← put the .py and .npy files here
│   ├── big_icnn_model.py
│   ├── big_icnn_train_v2.py
│   └── ...
└── kyber.py                ← already here from the submodule
```

Copy the scripts into place:
```bash
cp *.py message-recovery-attack-on-ml-kem/icnn_experiments/
cp *.npy message-recovery-attack-on-ml-kem/icnn_experiments/
```

---

## Step 3 — Get the datasets

The datasets are not included here. Download them from Zenodo and place them in `message-recovery-attack-on-ml-kem/datasets/`:

> **Zenodo:** https://zenodo.org/records/18702355

| File | What it's for |
|------|--------------|
| `kem_dec_unprotected_8.h5` | Training the CNN models |
| `kem_dec_unprotected_8_attack.h5` | Evaluating the attack on unseen secrets |
| `sec_decoder_masked.h5` | Phase 2: masked implementation (coming soon) |

---

## Step 4 — Build the Kyber library

The scripts call the Kyber-512 C library to compute message labels. Build it once:

```bash
cd message-recovery-attack-on-ml-kem/kyber/ref
make
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
cd ../../..
```

---

## Step 5 — Install Python dependencies

```bash
pip install torch numpy h5py matplotlib scalib
```

---

## Step 6 — Run the scripts

All scripts should be run from inside `icnn_experiments/`:

```bash
cd message-recovery-attack-on-ml-kem/icnn_experiments
```

**Compute CNN-optimized POIs** (or skip — pre-computed `.npy` file is already included):
```bash
python create_icnn_poi.py
```

**Train and attack the Single-Bit CNN across all 8 bit positions:**
```bash
python train_attack_all_bits.py
```

**Train the Multi-Bit CNN** (use GPU — this is slow on CPU):
```bash
python big_icnn_train_v2.py
```

**Run the attack on unseen secrets:**
```bash
python big_icnn_attack_v2.py
```

**Generate plots:**
```bash
python plot_poi_comparison.py
python plot_single_bit_results.py
```

---

## Attribution

This work builds on the open-source attack framework by Patrik Dobias and Lukas Malina:  
https://github.com/paprikadobi/message-recovery-attack-on-ml-kem

The `kyber.py` module and dataset structure are reused from that repository. Everything in `icnn_experiments/` is original work from this internship.

---

## Hardware used

- **Training (CPU runs):** Linux VM, Intel CPU
- **Training (GPU runs):** Google Colab, NVIDIA Tesla T4 (14.6 GB VRAM)
- **Target device:** Sakura-X FPGA board, 3 MHz clock
- **Oscilloscope:** Keysight MSOS104A, 200 MS/s sampling rate
