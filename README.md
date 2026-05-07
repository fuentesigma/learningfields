# A non-abelian discrete Stokes law for $\mathsf S_N$-valued connections on $2$-complexes

Code accompanying the paper

All scripts are self-contained: run from this directory and outputs are
written to `./figs/`. No external data is required; the small cache
`ensemble_cache_seed0_N10_nev30.npz` is shipped to avoid a ~5 min cold
start. Delete it to force regeneration from scratch.

## Requirements

Python ≥ 3.10 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Tested with CPU-only PyTorch; no GPU required. All scripts use the
non-interactive `Agg` matplotlib backend.

## Figure → script mapping

The paper cites five figures. Each is produced by a single command:

| Paper figure              | File                            | Producer              |
|---------------------------|---------------------------------|-----------------------|
| Fig. `fig:body`           | `figs/body_figure.pdf`          | `python fig_body.py`  |
| Fig. `fig:results`        | `figs/fig_loop_diagnostics.pdf` | `python fig_loop.py`  |
| Fig. `fig:sanity`         | `figs/sanity_checks.pdf`        | `python fig_sanity.py`|
| Fig. `fig:gap-diagnostics`| `figs/fig_gap_diagnostics.pdf`  | `python fig_gap.py`   |
| Fig. `fig:char-stokes`    | `figs/fig_char_stokes.pdf`      | `python fig_stokes.py`|

## Run order

The figure scripts depend on intermediate `.npz` caches produced by the
compute scripts. To reproduce everything from zero, run in this order:

```bash
python gap_distribution.py       # diagnostic; emits figs/gap_distribution.pdf
python gap_sweep.py              # produces figs/gap_enforced_sweep.npz
python character_spectrum.py     # produces figs/character_spectrum.npz
python topological_signature.py  # reads gap_enforced_sweep.npz

python fig_loop.py
python fig_gap.py
python fig_stokes.py
python fig_body.py
python fig_sanity.py
```

Total wall time with the shipped ensemble cache: ~5–10 min on a recent
laptop CPU. Without the cache, add ~5 min for ensemble training on the
first compute script that runs.

## File inventory

Shared modules:

- `core.py`  — ensemble training, transport operators, loop holonomy
- `style.py` — shared matplotlib style and helpers

Compute scripts (emit `.npz` caches; also emit their own diagnostic PDFs):

- `gap_distribution.py`
- `gap_sweep.py`
- `character_spectrum.py`
- `topological_signature.py`

Figure scripts (consume the caches, emit paper figures):

- `fig_body.py`
- `fig_loop.py`
- `fig_gap.py`
- `fig_stokes.py`
- `fig_sanity.py`

Cache:

- `ensemble_cache_seed0_N10_nev30.npz` — pre-computed ensemble
  predictions and Jacobians on the 30×30 grid, seed 0, 10 models.
  Regenerated automatically if missing.

## Determinism

All seeded entry points use `seed0=0`, `n_models=10`, `n_eval=30` by
default. Outputs are deterministic modulo the non-determinism of
PyTorch's BLAS on different platforms; the reported statistics match
to at least three significant figures on x86_64 Linux and arm64 macOS.
