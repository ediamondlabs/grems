# Installation

## Requirements

- Linux or macOS (developed on Debian 12, Python 3.11).
- Python 3.9 or newer. The released results were produced with Python 3.11.
- About 1 GB of disk space (most of it is the checkpoints).
- A GPU is only needed for re-training. Evaluation, the demo and all baselines run on CPU.

| Package | Tested version | Used for |
|---------|---------------|----------|
| torch | 2.10 | actor critic, EnSplit DDQN |
| gymnasium | 1.2 | environment interface |
| numpy | 2.4 | everything |
| networkx | 3.6 | topologies and routing |
| matplotlib | 3.10 | topology plots and the demo |
| wandb | 0.25 (optional) | training curves when `--wandb-project` is given |

## Steps

```bash
git clone <repository-url> GReMS
cd GReMS

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use conda:

```bash
conda create -n grems python=3.11 -y
conda activate grems
pip install -r requirements.txt
```

On a GPU machine install the PyTorch build that matches your CUDA version first, following <https://pytorch.org/get-started/locally/>, then run `pip install -r requirements.txt`.

## Unpack the network topologies

The train and test topologies are stored as small compressed archives in `data/networks/`.
Extract them once:

```bash
bash scripts/setup_networks.sh            # all families
bash scripts/setup_networks.sh rgg        # only random geometric graphs
```

This creates `variable_networks_rgg/`, `variable_networks_ws/`, `variable_networks_er/`, `variable_networks/` (Barabasi-Albert, m = 2) and the test only sets `variable_networks_ba_m1/`, `_m3/`, `_m4/` next to the code.

!!! warning "Do not regenerate the topologies if you want paper numbers"
    `generate_variable_networks.py` is seeded, but the shipped networks were created by earlier revisions of the generator.
    Regenerating gives different graphs and therefore slightly different numbers.
    Use the archives for reproduction and the generator only for new experiments.

## Check the installation

```bash
python -c "import torch, gymnasium, networkx; print(torch.__version__, gymnasium.__version__, networkx.__version__)"
python -c "import config; print(config.MAX_NODES, config.NUM_LAYERS)"      # prints: 20 21
```

Then continue with the [Quick start](quickstart.md).

## Building this documentation

```bash
pip install -r requirements-docs.txt
mkdocs serve        # live preview at http://127.0.0.1:8000
mkdocs build        # static site in site/
```
