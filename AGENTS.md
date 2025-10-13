# Repository Guidelines

## Project Structure & Module Organization
Core Python modules live in `ktransformers/` (scheduler logic, server entrypoints, optimization rules). CUDA/C++ kernels and build helpers sit in `csrc/`, with compiled artifacts landing under `build/`. Reference models, prompts, and evaluation sets are versioned in `data/` and `ktransformers/tests/`. Documentation and specs reside in `docs/` and `specs/`, while automation scripts live in `scripts/` (for example `scripts/initial_build.sh`). Treat `third_party/` as read-only vendor code.

## Build, Test, and Development Commands
Install editable dependencies with `pip install -e .` from `/workspace` to make the Python package importable. Use `bash scripts/initial_build.sh` for a full CUDA toolchain bootstrap, or `bash scripts/quick_build.sh` for incremental rebuilds after minor changes. Launch the FastAPI server locally via `python -m ktransformers.server.main --help` to inspect available backends; documenting any new CLI flags keeps the README in sync.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents and descriptive snake_case for Python functions, PascalCase for classes, and UPPER_SNAKE_CASE for constants. The repo uses `black` with a 120-character line length (`black ktransformers scripts`). Co-locate Triton or CUDA kernels beside their Python wrappers and mirror filenames (`foo_kernel.cu` ↔ `foo_kernel.py`) for discoverability. Prefer type hints for new interfaces and keep module-level docstrings brief and actionable.

## Testing Guidelines
Unit and integration suites live under `ktransformers/tests/` (`unit/`, `integration/`, `profiling/`). Run targeted suites with `pytest ktransformers/tests/unit` before opening a PR, and execute hardware-sensitive checks (`pytest ktransformers/tests/integration`) when modifying scheduler or kernel logic. Performance regressions should be validated using `python ktransformers/tests/test_speed.py --api_url ...` and the relevant model assets in `data/`. Aim to keep new code covered by at least one automated test or documented benchmark.

## Commit & Pull Request Guidelines
Git history favors concise, imperative subject lines, optionally scoped (`docs: refresh deployment notes`). Group related changes into single commits and include context in the body when touching kernels or performance-critical paths. PRs should summarize the change, list validation steps (`pytest`, benchmarks, or manual runs), and link to specs in `specs/` or issues when applicable. Attach logs or screenshots for UI-leaning updates (CLI output, benchmark tables) so reviewers can verify results quickly.
