# avogadro-auto3d

An Avogadro 2 plugin that generates low-energy conformers with
[Auto3D](https://github.com/isayevlab/Auto3D_pkg), optimizing and ranking them
with a neural network potential (AIMNet2 or ANI). The GPU is used automatically
when CUDA is available through PyTorch, otherwise it runs on the CPU.

## Menu

Extensions → Auto3D → Generate Conformers…

## Options

- **Conformers** — number of low-energy conformers to keep (Auto3D `k`,
  default 10).
- **NNP Method** — neural network potential used to optimize and rank
  conformers: `AIMNET` (default, AIMNet2), `ANI2x`, or `ANI2xt`.
- **Enumerate tautomers** — also generate and rank tautomers (default off).

Each returned conformer is tagged with its relative `<energy>` (kJ/mol, relative
to the lowest-energy structure) so Avogadro can plot the conformer energies.

Avogadro supplies a fully defined 3D structure, so Auto3D is run with
stereoisomer enumeration disabled — the configuration of the input is always
preserved. (Tautomers, when enabled, share the input's molecular formula, so
they are returned in the same conformer set.)

## Notes

Auto3D 3.x downloads its AIMNet2 model into `~/.cache/aimnet` on first use, so
the first run requires network access. Set `AIMNET_CACHE_DIR` to override the
cache location.

## CUDA

PyTorch comes from conda-forge, and the manifest declares CUDA-capable variants
of `linux-64` and `win-64` ahead of the plain CPU ones:

```toml
platforms = [
  { name = "linux-cuda", platform = "linux-64", cuda = "12.0" },
  { name = "win-cuda", platform = "win-64", cuda = "12.0" },
  { name = "linux-cpu", platform = "linux-64" },
  { name = "win-cpu", platform = "win-64" },
  "osx-arm64",
]
```

All five variants are solved into the single `pixi.lock`, and `pixi install`
picks one at install time from the `__cuda` virtual package the host reports —
`pytorch-gpu` where a CUDA driver is present, `pytorch-cpu` otherwise. There is
still only one `default` environment, so nothing has to be selected by hand.
macOS uses the Metal (MPS) backend from the same plain `pytorch` build.

Auto3D itself is only on PyPI: conda-forge's `auto3d` is still 2.3.0 (August
2024), which predates the `Auto3DOptions` API, and `aimnet` has no conda-forge
package at all.

Run `pixi workspace platform list` to see which variant matches the current
machine.

## Development

```sh
pixi install
# Smoke test from a saved Avogadro JSON payload:
pixi run python -m avogadro_auto3d conformers < sample.json
```

## Suggestions

Please submit suggestions, bugs, etc. on GitHub:
https://github.com/ghutchis/avogadro-auto3d/issues
