# Contributing to torchquad

Thanks for your interest in improving torchquad! The full guide lives in the
[contributing documentation](https://torchquad.readthedocs.io/en/main/contributing.html);
this file is the quick reference GitHub shows when you open a pull request.

## Branching model

torchquad uses a two-branch model:

- **`develop`** — the integration branch. **All feature and fix PRs target
  `develop`.**
- **`main`** — release branch. It only ever receives releases (via a
  `develop` → `main` merge) and the occasional documentation hotfix for the
  currently released version.

So, unless you are fixing docs for the *current release*, branch off `develop`
and open your PR against `develop`. Releases are cut by merging `develop` into
`main`, tagging, and publishing; `main` therefore stays a subset of `develop`.

## Before you push

CI enforces formatting, linting, docstrings, dead-code, and the full test suite.
Run these locally first (see `CLAUDE.md` / the CI docs for the full list):

```bash
ruff format .
ruff check .
pydoclint torchquad/
cd tests/ && pytest
```

Or install the hooks once with `pre-commit install` so they run automatically.

## Review

PRs are reviewed against `REVIEW.md` (correctness-first, backend-agnostic,
vectorize-or-fail, fail-hard). Skimming it before you open a PR helps your
change land faster.
