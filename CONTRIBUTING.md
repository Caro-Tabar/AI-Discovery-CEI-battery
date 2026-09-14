# Contributing to CEI-Scout

CEI-Scout is currently a solo portfolio project, but contributions follow a
review-ready workflow to preserve code quality and reproducibility.

## Branch workflow

The initial repository scaffold is committed directly to `main`. After that:

1. Start from the current `main` branch.
2. Create one short-lived branch for each focused change.
3. Use a descriptive branch prefix:
   - `feature/` for new functionality
   - `fix/` for corrections
   - `docs/` for documentation
   - `test/` for test changes
   - `chore/` for maintenance
4. Open a pull request and wait for CI to pass before merging.
5. Do not commit directly to `main` after the initial scaffold.

## Required checks

Before committing, run:

```bash
pre-commit run --all-files
mypy
python -m pytest
