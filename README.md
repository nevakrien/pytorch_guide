# pytorch_guide

A short, practical guide to getting started with PyTorch basics. This focuses on
creating tensors, working in a virtual environment, and understanding how
operations behave across shapes.

## 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch
```

Quick sanity check:

```bash
python -c "import torch; print(torch.__version__)"
```

Deactivate when you're done:

```bash
deactivate
```

## 2) run

Run the full, runnable examples with:

```bash
python main.py
```

The `main.py` script walks through tensor creation, shapes, operations,
matmul with different shapes, and indexing.

