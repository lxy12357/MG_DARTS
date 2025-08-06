# MG-DARTS: Multi-Granularity Differentiable Architecture Search

This repository contains the official implementation of **MG-DARTS (Multi-Granularity Differentiable Architecture Search)**, a unified differentiable neural architecture search framework to identify efficient neural networks from scratch by jointly exploring multiple levels of architectural granularity, including:

- **Operation-level**
- **Filter-level**
- **Weight-level**


## Search Process

To run the architecture search process on CIFAR datasets, execute:

```bash
python cifar_search/train_search.py
```

---

## Evaluation Process

To evaluate the architectures discovered by MG-DARTS, please follow the instructions below:

CIFAR-10 (large architecture)
```bash
python cifar_train/train.py --arch MG_DARTS --mask_path ../masks/mask_cifar10_darts.npy
```

CIFAR-10 (small architecture)
```bash
python cifar_train/train.py --arch MG_DARTS_small --init_channels 24 --mask_path ../masks/mask_cifar10_darts_small.npy
```

CIFAR-100
```bash
python cifar_train/train_cifar100.py --arch MG_DARTS --mask_path ../masks/mask_cifar10_darts.npy
```


