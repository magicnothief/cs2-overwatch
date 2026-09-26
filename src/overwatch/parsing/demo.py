"""Parse a CS2 .dem file into the canonical tables — in a process of its own.

`parse_demo()` spawns `python -m overwatch.parsing.worker`, waits for it, and
reads the tables back from a temporary directory. The reason is in worker.py:
the demo is a file a stranger wrote, the parser is a Rust extension, and a
memory-safety bug there is not something `except BaseException` can hold. The
process boundary is the fix.

What is left in this process is a parquet read of files our own child wrote,
whose *contents* still come from the demo. That is a far smaller surface than a
protobuf parser, not a zero one. See MAG-8 for the residual risk.
"""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from overwatch.parsing.types import DemoParseError, ParsedMatch
from overwatch.parsing.worker import read_match

#: How long one demo may take to parse before the worker is killed. A 26 MB
#: demo parses in under a second; this is a hang, not a slow machine.
PARSE_TIMEOUT = 300.0

#: The environment the child keeps. Everything else — whatever a shell exported,
#: whatever the API server was started with — stays in the parent.
KEEP_ENV: tuple[str, ...] = (
    "APPDATA",
    "HOME",
    "LD_LIBRARY_PATH",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "PYTHONHOME",
    "PYTHONPATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "VIRTUAL_ENV",
)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file so the same demo is never parsed (or labeled) twice."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _worker_command(demo: Path, out_dir: Path) -> list[str]:
    """The child's whole world: the demo to read and the directory to write."""
    return [
        sys.executable,
        "-m",
        "overwatch.parsing.worker",
        str(demo),
        str(out_dir),
    ]


def _child_env(timeout: float) -> dict[str, str]:
    env = {name: os.environ[name] for name in KEEP_ENV if name in os.environ}
    env["OVERWATCH_PARSE_CPU_SECONDS"] = str(int(timeout))
    # The memory cap is the worker's own (see DEFAULT_MEMORY_MB); forward an
    # override so a machine that needs a different one can say so.
    memory = os.environ.get("OVERWATCH_PARSE_MEMORY_MB")
    if memory is not None:
        env["OVERWATCH_PARSE_MEMORY_MB"] = memory
    return env


def _readable(returncode: int, errors: str) -> str:
    """Why the worker died, in one line a reviewer can act on."""
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        return f"the parser was killed by {name}: the demo is damaged or hostile"
    last = next((line for line in reversed(errors.splitlines()) if line.strip()), "")
    # A traceback names site-packages, which on most machines names the home
    # directory. The page shows this string; the home directory stays out of it.
    last = last.replace(str(Path.home()), "~").strip()[:200]
    return f"the parser exited {returncode}" + (f": {last}" if last else "")


def _run_worker(demo: Path, out_dir: Path, timeout: float) -> None:
    command = _worker_command(demo, out_dir)
    try:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_child_env(timeout),
            text=True,
            errors="replace",
        )
    except OSError as exc:
        raise DemoParseError(f"could not start the parser: {exc.strerror}") from exc

    try:
        _, errors = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        try:
            # Reap it, so a timeout leaves no orphan behind. SIGKILL cannot be
            # refused, but a pipe some grandchild inherited could still hold
            # this open: do not trade a hung parse for a hung server.
            child.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
        raise DemoParseError(
            f"the parser did not finish within {timeout:.0f}s and was stopped"
        ) from None
    if child.returncode != 0:
        raise DemoParseError(_readable(child.returncode, errors))


def parse_demo(
    path: str | Path, *, hash_file: bool = True, timeout: float = PARSE_TIMEOUT
) -> ParsedMatch:
    """Parse a .dem file into canonical tick and event tables.

    Args:
        path: path to the .dem file.
        hash_file: compute a sha256 of the demo (set False for speed in tests).
        timeout: seconds the worker gets before it is killed.

    Returns:
        A ParsedMatch whose ticks hold TICK_COLUMNS, sorted by (player_id, tick).

    Raises:
        DemoParseError: the worker crashed, timed out, or wrote no tables.
    """
    path = Path(path)
    if not path.is_file():
        raise DemoParseError(f"no demo to parse at {path.name}")

    with tempfile.TemporaryDirectory(prefix="overwatch-parse-") as scratch:
        out_dir = Path(scratch)
        _run_worker(path, out_dir, timeout)
        match = read_match(out_dir)

    match.meta["sha256"] = file_sha256(path) if hash_file else None
    return match
