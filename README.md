# KYBER-ICNN: Deep Learning-Based Message Recovery Against ML-KEM

> **Research Internship Project** — Radboud University, Nijmegen, Netherlands  
> Erasmus Mundus Joint Master's in Cybersecurity (CYBERUS), Université Bretagne Sud  
> Supervised by Ms. Azade Rezaeezade and Prof. Lejla Batina

---

## Overview

This repository contains the deep learning code developed during a research internship as part of a larger project on *Power Side-Channel Leakage Evaluation of Unified Protected ML-DSA and ML-KEM Hardware Implementations*.

This work specifically targets the **Masked Message Decoder** component of the ML-KEM (Kyber-512) decapsulation pipeline, investigating whether Convolutional Neural Networks can recover secret message bits from a single power trace, both on an unprotected and a first-order masked FPGA implementation.

The work proceeds in two phases:

- **Phase 1 (this repository):** Single-trace message bit recovery on the unprotected implementation, establishing a CNN-based methodology and baseline.
- **Phase 2 (ongoing):** Extension to the first-order masked implementation using the same CNN architecture.

---

## Key Results

| Method | Single-trace accuracy | Traces required |
|--------|----------------------|-----------------|
| LDA (Dobias & Malina, 2026) | ~0.37% (below random) | 4–16 |
| SimpleCNN (this work) | 64.00% (best bit) | **1** |
| Multi-bit CNN (this work) | 60.25% (256 bits) | **1** |

- **31 of 32** message bytes recovered above random chance on 200 unseen secrets
- Training–attack gap of only **0.13%**, confirming generalization

---

## Dependencies and Attribution

This work builds upon the open-source attack framework released by Dobias & Malina (EEICT 2026):

> **Message Recovery Attack on ML-KEM**  
> Patrik Dobias, Lukas Malina  
> https://github.com/paprikadobi/message-recovery-attack-on-ml-kem

The researcher's repository is included as a Git submodule. The following components are reused directly:

- `kyber.py` — Kyber-512 software implementation including `extract_msg()`
- Dataset structure and format (`.h5` files)
- LDA POI formula and `main.py` attack pipeline (reproduced for baseline)

All CNN architectures, POI selection scripts, training pipelines, and attack evaluation scripts in `icnn_experiments/` are **original contributions** of this work.

---

## Datasets

Datasets are **not included** in this repository due to size constraints. They are publicly available at:

> Zenodo Record: [https://zenodo.org/records/18702355](https://zenodo.org/records/18702355)

The following datasets are used:

| File | Size | Purpose |
|------|------|---------|
| `kem_dec_unprotected_8.h5` | ~1.2 GB | Training (100,000 traces) |
| `kem_dec_unprotected_8_attack.h5` | ~600 MB | Attack evaluation (51,200 traces) |
| `sec_decoder_masked.h5` | ~10 GB | Masked implementation (Phase 2) |

Place all datasets in a `datasets/` folder at the same level as `icnn_experiments/`:

```
your_working_directory/
├── datasets/
│   ├── kem_dec_unprotected_8.h5
│   ├── kem_dec_unprotected_8_attack.h5
│   └── sec_decoder_masked.h5
└── icnn_experiments/
```

---

## Setup

### 1. Clone with submodule

```bash
git clone --recurse-submodules https://github.com/aryastark08/KYBER-ICNN-Side-Channel-Analysis_Masked_Implementation.git
cd KYBER-ICNN-Side-Channel-Analysis_Masked_Implementation
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Build the Kyber library

```bash
cd message-recovery-attack-on-ml-kem/kyber/ref
make
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
```

### 3. Install Python dependencies

```bash
pip install torch numpy h5py matplotlib scalib
```

### 4. Download datasets

Download from Zenodo and place in `datasets/` as described above.

---

## Running the Code

All scripts should be run from inside the `icnn_experiments/` directory:

```bash
cd icnn_experiments/
```

### Step 1: Compute CNN-optimized POIs

```bash
python create_icnn_poi.py
# Output: cnn_optimal_pois.npy
```

### Step 2: Train SimpleCNN (single-bit, proof of concept)

```bash
python train_attack_all_bits.py
# Trains 8 independent models (one per bit)
# Output: all_bits_results.npy, weights_bit{N}.pt
```

### Step 3: Train ICNN (multi-bit, all 256 bits)

```bash
# Recommended: run on GPU (Google Colab or local CUDA)
python big_icnn_train_v2.py
# Output: big_icnn_best.pt, big_icnn_byte_accuracies.npy
```

### Step 4: Attack evaluation

```bash
python big_icnn_attack_v2.py
# Output: icnn_attack_results.npy
```

### Step 5: Generate plots

```bash
python plot_poi_comparison.py
python plot_single_bit_results.py
```

---

## Hardware Used

| Component | Specification |
|-----------|--------------|
| Training device (Phase 1 CPU) | Linux VM, Intel CPU |
| Training device (Phase 1 GPU) | Google Colab, NVIDIA Tesla T4 (14.6 GB) |
| Target FPGA | Sakura-X board, 3 MHz |
| Oscilloscope | Keysight MSOS104A, 200 MS/s |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{nair2026kyber,
  author = {Nair, Aishwarya Jeevan},
  title  = {Deep Learning-Based Message Recovery Against ML-KEM:
            From Unprotected to Masked Implementations},
  year   = {2026},
  note   = {Research Internship Report, Radboud University, 
            Université Bretagne Sud (CYBERUS)}
}
```

And the baseline work this builds upon:

```bibtex
@inproceedings{dobias2026message,
  author    = {Dobias, Patrik and Malina, Lukas},
  title     = {Message Recovery Attack on ML-KEM},
  booktitle = {EEICT 2026},
  year      = {2026}
}
```

---

## License

This repository contains original research code developed during an academic internship.  
The submodule `message-recovery-attack-on-ml-kem` retains its original license from the respective authors.

---

*Part of the broader project: Power Side-Channel Leakage Evaluation of Unified Protected ML-DSA and ML-KEM Hardware Implementations — Radboud University, 2026.*
