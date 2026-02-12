# On Conservative Learning Fields

J. Fuentes Aguilar

This repo contains the code for the paper. It trains an MLP ensemble on a torus, builds entropic and exact transport couplings between jet-enriched fibres, and composes Wilson loops to get holonomy diagnostics.

## Setup

```
pip install -r requirements.txt
```

Needs Python 3.8+, PyTorch, NumPy, SciPy, Matplotlib. See `requirements.txt` for versions.

## Running

Everything at once:

```
python run_experiments.py
```

Or individually:

```
python task_1_regime_bridging.py    # entropic vs Hungarian
python task_2_assignment_limit.py   # epsilon sweep
python task_3_gauge_invariance.py   # relabelling invariance
```

Tests: `python test_framework.py`

## Files

- `framework.py` — all the core stuff: torus geometry, Sinkhorn, jet metrics, holonomy
- `run_experiments.py` — runs tasks 1-3 and an epsilon sweep
- `task_1_regime_bridging.py` — Sinkhorn vs Hungarian on the same torus
- `task_2_assignment_limit.py` — coupling concentrates to permutation as epsilon goes to zero
- `task_3_gauge_invariance.py` — checks holonomy is invariant under random relabelling
- `test_framework.py` — 8 integration tests, run these first

Figures go to `paper/figs/`.

## License

MIT

-[O_o]- j.fuentesaguilar
