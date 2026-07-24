# Deep Learning-Based Message Recovery of the ML-KEM Hardware Decoder

**Research Internship — Radboud University, Nijmegen, Netherlands**  
Supervised by Ms. Azade Rezaeezade and Prof. Lejla Batina

---

## What is this?

This repository contains the code developed during a research internship investigating whether deep learning can recover secret message bits from power traces of the ML-KEM (Kyber-512) hardware decoder, a component of a post-quantum cryptographic implementation running on an FPGA.

The work proceeds in two phases:
- **Phase 1 (this repo):** Attack the unprotected hardware decoder using CNN-based single-trace power analysis
- **Phase 2 (in progress):** Extend the attack to a first-order masked implementation

The key finding is that a CNN achieves **60.25% single-trace bit-level accuracy** across 200 unseen secrets compared to LDA's 0.37% on a single trace; demonstrating that deep learning can exploit leakage where classical template attacks completely fail.

| Model | Single-trace accuracy | Traces needed |
|---|---|---|
| LDA template attack (reproduced) | ~0.37% | 4–16 |
| Single-Bit CNN (this work) | 64% best bit, 59.6% mean | **1** |
| Multi-Bit CNN (this work) | 60.25% across all 256 bits | **1** |

This work builds directly on the attack framework by Dobias and Malina (EEICT 2026), included here as a git submodule.

---

## Repository structure

```
├── single_bit_cnn_model.py        # Single-Bit CNN architecture
├── single_bit_dataset_loader.py   # Dataset loader for Single-Bit CNN
├── single_bit_train.py            # Train Single-Bit CNN on one bit position
├── single_bit_attack.py           # Evaluate Single-Bit CNN on attack dataset
├── single_bit_train_attack_all.py # Train + attack all 8 bit positions

├── multi_bit_cnn_model.py         # Multi-Bit CNN architecture (17.9M params)
├── multi_bit_dataset_loader.py    # Dataset loader for Multi-Bit CNN
├── multi_bit_train.py             # Train Multi-Bit CNN (GPU recommended)
├── multi_bit_attack.py            # Evaluate Multi-Bit CNN on attack dataset

├── compute_cnn_pois.py            # Compute CNN-optimized Points of Interest
├── plot_poi_comparison.py         # Plot LDA POIs vs CNN POIs
├── plot_single_bit_results.py     # Plot per-bit accuracy results

├── cnn_optimal_pois.npy           # Pre-computed CNN POIs (534 per byte)
├── single_bit_results.npy         # Single-Bit CNN results (all 8 bits)
├── multi_bit_train_results.npy    # Multi-Bit CNN per-byte training accuracy
├── multi_bit_attack_results.npy   # Multi-Bit CNN attack results

├── Kyber_ICNN.ipynb               # Main Colab notebook (recommended entry point)
├── message-recovery-attack-on-ml-kem/  # Submodule: original attack framework
└── README.md
```

> **Note:** Model weight files (`.pt`) are not included because they are too large and specific to a particular training run. Run the training scripts to reproduce them.

---

## Requirements

- Python 3.8+
- A Linux environment (all scripts were developed and tested on Linux/Colab)
- GPU strongly recommended for Multi-Bit CNN training (Google Colab T4 used in this work)
- The datasets from Zenodo — see Step 3 below

---

## Reproduction guide

### Step 1 — Clone the repository

```bash
git clone --recurse-submodules https://github.com/aryastark08/KYBER-ICNN-Side-Channel-Analysis_Masked_Implementation.git
cd KYBER-ICNN-Side-Channel-Analysis_Masked_Implementation
```

The `--recurse-submodules` flag pulls in the original attack framework automatically. If you forgot it:

```bash
git submodule update --init --recursive
```

---

### Step 2 — Set up the directory structure

The scripts expect to run from inside `message-recovery-attack-on-ml-kem/icnn_experiments/` and look for datasets at `../datasets/`. Set it up like this:

```
message-recovery-attack-on-ml-kem/
├── datasets/
│   ├── kem_dec_unprotected_8.h5         ← training dataset
│   ├── kem_dec_unprotected_8_attack.h5  ← attack evaluation dataset
│   └── sec_decoder_masked.h5            ← masked dataset (Phase 2)
├── icnn_experiments/                    ← scripts go here
│   ├── multi_bit_train.py
│   ├── cnn_optimal_pois.npy
│   └── ...
└── kyber.py                             ← from submodule
```

Copy the scripts and pre-computed results into place:

```bash
cp *.py message-recovery-attack-on-ml-kem/icnn_experiments/
cp *.npy message-recovery-attack-on-ml-kem/icnn_experiments/
```

---

### Step 3 — Download the datasets

The datasets are available from Zenodo and should be placed in `message-recovery-attack-on-ml-kem/datasets/`:

> **Zenodo:** https://zenodo.org/records/18702355

| File | Purpose |
|---|---|
| `kem_dec_unprotected_8.h5` | Training (100,000 traces) |
| `kem_dec_unprotected_8_attack.h5` | Attack evaluation (200 unseen secrets) |
| `sec_decoder_masked.h5` | Phase 2 — masked implementation |

---

### Step 4 — Build the Kyber library

The training scripts call the Kyber-512 C library to compute message labels from raw inputs. Build it once:

```bash
cd message-recovery-attack-on-ml-kem/kyber/ref
make
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
cd ../../..
```

---

### Step 5 — Install Python dependencies

```bash
pip install torch numpy h5py matplotlib scalib
```

---

### Step 6 — Run the scripts

All scripts should be run from inside `icnn_experiments/`:

```bash
cd message-recovery-attack-on-ml-kem/icnn_experiments
```

**Optional — recompute CNN-optimized POIs:**  
A pre-computed `cnn_optimal_pois.npy` is already included. Only run this if you want to recompute from scratch:
```bash
python compute_cnn_pois.py
```

**Single-Bit CNN — train and attack all 8 bit positions:**
```bash
python single_bit_train_attack_all.py
```

**Multi-Bit CNN — train (GPU recommended):**
```bash
python multi_bit_train.py
```

**Multi-Bit CNN — evaluate on attack dataset:**
```bash
python multi_bit_attack.py
```

**Generate plots:**
```bash
python plot_poi_comparison.py
python plot_single_bit_results.py
```

---

## Running on Google Colab

The recommended way to run training is via the included `Kyber_ICNN.ipynb` notebook, which handles the Kyber library setup, dataset loading, and GPU training automatically. Open it in Google Colab and run the cells in order.

---

## Attribution

This work builds on the open-source attack framework by Patrik Dobias and Lukas Malina:  
https://github.com/paprikadobi/message-recovery-attack-on-ml-kem

The `kyber.py` module (including `extract_msg()`) and the dataset structure are reused from that repository. All CNN architectures, POI selection scripts, training pipelines, and attack evaluation scripts in `icnn_experiments/` are original contributions of this internship.

---

## Hardware

| Component | Details |
|---|---|
| Training (CPU) | Linux VM, Intel CPU |
| Training (GPU) | Google Colab, NVIDIA Tesla T4 (14.6 GB) |
| Target device | Sakura-X FPGA board |
| Best model saved as | `multi_bit_best.pt` (not included — retrain to reproduce) |
