# 0009: Upper bounds on parse/serve dependencies installed at install time

- **Date:** 2026-09-26
- **Status:** accepted

## Context
`uv.lock` pins the whole dependency tree, but that lockfile is not what a
user's machine installs from. `install.sh` and `install.ps1` run
`uv tool install` against the built wheel, which resolves dependencies fresh
against `pyproject.toml`'s bounds at install time. `pyproject.toml` before
this decision carried lower bounds only (`fastapi>=0.141.1`, and eleven
more). Two users installing a week apart do not get the same dependency
set, and a break or a compromise in an upstream release reaches users
without any change on our side.

## Options considered
1. Leave lower bounds only. Cheapest, but a breaking (or compromised)
   upstream release ships to users the same day it's published, unfiltered.
2. Upper bounds on the direct dependencies that run at parse or serve
   time (`demoparser2`, `fastapi`, `uvicorn`, `pydantic`) and nothing else.
   Bounds these four; leaves the numeric stack (numpy, scipy, onnxruntime,
   polars, ...) alone, since those don't parse untrusted `.dem` input or
   serve the review page and a break there tends to be loud (import error,
   not silent misbehavior).
3. Ship a lock-derived constraints file (`uv export`) alongside the wheel
   and point `install.sh`/`install.ps1` at it with `--constraint`. Closest
   to reproducible, most maintenance: every `uv.lock` update needs a
   matching artifact republished and the installers need a flag added.

## Decision
Option 2. `demoparser2`, `fastapi`, `uvicorn` are capped at the next minor
(`>=X.Y.Z,<X.(Y+1).0`) since all three are pre-1.0 and, per semver, a minor
bump on a 0.x package is where breaking changes land. `pydantic` is capped
at `<3` since it's on a stable 2.x major.

## Consequences
A user's install can no longer silently pick up a breaking or compromised
release of the four packages that touch untrusted demo files or serve the
review page over localhost. The cost lands on us: a `demoparser2`, `fastapi`,
or `uvicorn` minor release, or a `pydantic` major, needs a `pyproject.toml`
bump and a fresh `uv.lock` before users can install it — `uv tool install`
will otherwise fail closed with an unsatisfiable-constraint error instead of
silently installing something older. That's the intended failure mode: loud
at install time, not silent in production.

The rest of the dependency tree (numpy, scipy, onnxruntime, polars, pillow,
embreex, trimesh, platformdirs, zstandard) keeps lower bounds only. If one
of those breaks in a way that isn't loud, revisit and widen this list.
