# pipeline: orchestration

**Purpose:** parse → L1 → L3 → L4, with L2 injected as a service. It also handles progress
callbacks, error isolation (one bad layer shouldn't kill the report), and caching by demo file
hash so the same demo is never parsed twice.

**Learn first:** dependency injection (pass layers in; don't import them globally), logging,
hashing files, CLI tools (`argparse` or `typer`).

**First exercise:** make `python -m overwatch analyze some.dem --out report.json` work
**before** writing any web code. The API is only a thin wrapper around this.

**Done when:** the CLI produces a report, rerunning it is fast (cache hit), and a failing layer
yields a partial report with the error recorded.
