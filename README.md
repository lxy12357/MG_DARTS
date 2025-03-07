# MGAS: Multi-Granularity Architecture Search

This repository contains the official implementation of **MGAS (Multi-Granularity Architecture Search)**, a unified framework for efficient neural architecture search across multiple granularity levels, including:

- **Operation-level search**
- **Kernel-level search**
- **Weight-level search**


## Search Process

To run the architecture search process on CIFAR datasets, execute:

```bash
python cifar_search/train_search.py
```

---

## Evaluation Process

To evaluate the architecture discovered by MGAS, execute:

```bash
python cifar_train/train.py --arch MGAS
```

