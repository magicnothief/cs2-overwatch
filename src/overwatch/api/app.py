"""The web interface's backend: upload a demo, follow the analysis, read reports.

    GET  /                          the page (overwatch/web/index.html)
    GET  /api/status                what is available: scorer, judge, busy
    POST /api/analyses?name=&judge= the demo as the raw request body; starts a job
    GET  /api/analyses/{id}         a job's progress
    GET  /api/reports               every saved report, newest first
    GET  /api/reports/{id}          one report (pipeline.MatchReport as JSON)
    GET  /api/radars/{map}          where a map's radar sits in world coordinates
    GET  /radars/{map}.png          the radar itself (training/rules/render_radars.py)

Local only (127.0.0.1, no accounts). An uploaded demo is stored under its content
hash, so the name a browser sends is only ever displayed, never used as a path.

"Local only" is a binding address, not an access control: every page the user
has open can reach 127.0.0.1, and a hostile DNS name can be made to resolve to
it. So `only_this_machine` below checks two headers on every request. The Host
header must name this machine, which stops DNS rebinding — an attacker's page
served from `evil.example` is same-origin with this server once that name
resolves to 127.0.0.1, and would otherwise read every report. A request that
changes something must carry no Origin header (a terminal, the CLI) or a local
one (this page); an Origin from anywhere else is another site driving the
user's browser, and is refused.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import threading
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from overwatch import paths, updates
from overwatch.api.jobs import Runner
from overwatch.demos import (
    DEMO_MAGIC,
    DemoTooLarge,
    demo_folders,
    list_demos,
    packing,
    resolve,
    unpack,
)
from overwatch.layers.l4_judge.server import nvidia_cuda_major
from overwatch.maps.steam import find_cs2_maps, maps_folder
from overwatch.settings import Settings, save_settings

WEB = Path(__file__).resolve().parents[1] / "web"

#: Competitive demos run 50-150 MB, tournament ones up to ~500 MB.
MAX_DEMO_BYTES = 2 * 1024**3
_REPORT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAP_NAME = re.compile(r"^[a-z0-9_]+$")

#: The names this machine answers to. Any other Host is a name that was pointed
#: at 127.0.0.1 to reach this server from a site (DNS rebinding).
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
#: Methods that change nothing, so an Origin from elsewhere cannot do harm with
#: them: the same-origin policy already keeps the answer from being read.
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
#: Sent with the page. 'unsafe-inline' for styles only: app.js sets a style
#: attribute on the radar's gradient stops; scripts stay 'self'.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


def _hostname(value: str) -> str:
    """The host in a Host header, without its port and its IPv6 brackets."""
    if value.startswith("["):
        return value[1 : value.find("]")] if "]" in value else value[1:]
    return value.split(":", 1)[0]


def is_local_origin(origin: str) -> bool:
    """Is this Origin the page this server itself serves?"""
    split = urlsplit(origin)
    return split.scheme == "http" and (split.hostname or "") in LOCAL_HOSTS


class LocalChoice(BaseModel):
    folder: str
    name: str
    judge: str = "flagged"


def _stem(name: str) -> str:
    """A demo's report id: its name without the demo endings."""
    for suffix in (".gz", ".bz2", ".zst", ".dem"):
        name = name.removesuffix(suffix)
    return name


@functools.lru_cache(maxsize=256)
def _header_map(path: str, modified: float) -> str | None:
    try:
        from demoparser2 import DemoParser

        return DemoParser(path).parse_header().get("map_name") or None
    except KeyboardInterrupt:
        raise
    except BaseException:  # noqa: BLE001 - a truncated demo panics in the parser
        return None


def _map_of(path: Path) -> str | None:
    """The map a plain demo was played on, from its header (cached)."""
    if not path.name.endswith(".dem"):
        return None
    try:
        return _header_map(str(path), path.stat().st_mtime)
    except OSError:
        return None


def create_app(
    runner: Runner,
    *,
    uploads: Path = paths.UPLOADS,
    radars: Path = paths.RADARS,
    check_updates: bool = False,
) -> FastAPI:
    """`check_updates` asks GitHub for a newer release in the background (once a
    day at most, and only with the updates setting on); tests leave it off."""
    app = FastAPI(title="Overwatch review", docs_url=None, redoc_url=None)
    news: dict = {"update": None}

    def look_for_updates() -> None:
        if runner.settings().updates:
            release = updates.check()
            if release is not None:
                news["update"] = asdict(release) | {"command": updates.update_command()}

    if check_updates:
        threading.Thread(target=look_for_updates, daemon=True).start()

    @app.middleware("http")
    async def only_this_machine(request: Request, call_next):
        """Every request, before any route: the Host must be this machine, and
        anything that changes something must not come from another site."""
        if _hostname(request.headers.get("host", "")) not in LOCAL_HOSTS:
            return JSONResponse(
                {
                    "detail": "Open http://127.0.0.1 — this server answers to no "
                    "other name"
                },
                status_code=421,
            )
        origin = request.headers.get("origin")
        if (
            request.method not in READ_METHODS
            and origin is not None
            and not is_local_origin(origin)
        ):
            return JSONResponse(
                {
                    "detail": "Another site cannot start an analysis or change "
                    "your settings"
                },
                status_code=403,
            )
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Content-Security-Policy", CSP)
        return response

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(WEB / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        return runner.status() | {
            "version": updates.installed_version(),
            "update": news["update"],
        }

    @app.post("/api/analyses")
    async def analyse(
        request: Request, name: str = "demo.dem", judge: str = "flagged"
    ) -> dict:
        if judge not in ("flagged", "all", "none"):
            raise HTTPException(422, "judge must be flagged, all or none")
        if not runner.status()["scorer"]:
            raise HTTPException(
                503, "No scorer model yet: run training/scorer/train_final.py first"
            )
        uploads.mkdir(parents=True, exist_ok=True)
        partial = uploads / f".upload-{os.getpid()}-{id(request)}"
        digest, size, head = hashlib.sha256(), 0, b""
        with partial.open("wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_DEMO_BYTES:
                    fh.close()
                    partial.unlink(missing_ok=True)
                    raise HTTPException(413, "That file is larger than any CS2 demo")
                if len(head) < len(DEMO_MAGIC):
                    head += chunk[: len(DEMO_MAGIC) - len(head)]
                digest.update(chunk)
                fh.write(chunk)
        demo = uploads / f"{digest.hexdigest()[:16]}.dem"
        if head == DEMO_MAGIC:
            partial.replace(demo)
        elif packing(head):  # a packed demo (.dem.gz, .dem.bz2, .dem.zst)
            try:
                unpack(partial, demo)
            except DemoTooLarge as exc:
                raise HTTPException(413, str(exc)) from exc
            except (ValueError, OSError, EOFError) as exc:
                raise HTTPException(
                    422, "That is not a CS2 demo (.dem from CS2)"
                ) from exc
            finally:
                partial.unlink(missing_ok=True)
        else:
            partial.unlink(missing_ok=True)
            raise HTTPException(422, "That is not a CS2 demo (.dem from CS2)")
        return runner.submit(demo, Path(name).name, judge).public()

    def folders() -> dict[str, Path]:
        return demo_folders(runner.settings().cs2)

    @app.get("/api/demos")
    def local_demos() -> dict:
        """Demos in CS2's replays folder and in Downloads, newest first."""
        where = folders()
        return {
            "folders": {key: str(path) for key, path in where.items()},
            "demos": [
                {
                    "folder": d.folder,
                    "name": d.name,
                    "size": d.size,
                    "modified": d.modified,
                    "map": _map_of(where[d.folder] / d.name),
                    "reviewed": (runner.reports / f"{_stem(d.name)}.json").exists(),
                    "report_id": _stem(d.name),
                }
                for d in list_demos(where)
            ],
        }

    @app.post("/api/analyses/local")
    def analyse_local(choice: LocalChoice) -> dict:
        """Review a listed demo where it lies (a packed one is unpacked first)."""
        if choice.judge not in ("flagged", "all", "none"):
            raise HTTPException(422, "judge must be flagged, all or none")
        path = resolve(folders(), choice.folder, choice.name)
        if path is None:
            raise HTTPException(404, "No such demo in that folder")
        with path.open("rb") as fh:
            head = fh.read(len(DEMO_MAGIC))
        if head != DEMO_MAGIC:
            uploads.mkdir(parents=True, exist_ok=True)
            try:
                path = unpack(path, uploads / f"{_stem(choice.name)}.dem")
            except DemoTooLarge as exc:
                raise HTTPException(413, str(exc)) from exc
            except (ValueError, OSError, EOFError) as exc:
                raise HTTPException(
                    422, "That is not a CS2 demo (.dem from CS2)"
                ) from exc
        return runner.submit(path, choice.name, choice.judge).public()

    @app.get("/api/analyses/{job_id}")
    def job(job_id: str) -> dict:
        found = runner.jobs.get(job_id)
        if found is None:
            raise HTTPException(404, "No such analysis")
        return found.public()

    @app.get("/api/reports")
    def reports() -> list[dict]:
        out = []
        files = sorted(
            runner.reports.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for path in files:
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            players = data.get("players", [])
            out.append(
                {
                    "id": path.stem,
                    "demo": data.get("demo"),
                    "map_name": data.get("map_name"),
                    "kills": data.get("kills"),
                    "players": len(players),
                    "flagged": sum(1 for p in players if p.get("flagged")),
                    "analysed": path.stat().st_mtime,
                }
            )
        return out

    def machine() -> dict:
        chosen = runner.settings()
        found = find_cs2_maps(chosen.cs2)
        return {
            "cs2": chosen.cs2,
            "cs2_found": str(found) if found else None,
            "gpu": chosen.gpu,
            "updates": chosen.updates,
            "nvidia": nvidia_cuda_major() is not None,
            "home": str(paths.HOME),
        }

    @app.get("/api/settings")
    def get_settings() -> dict:
        """The saved choices, and where CS2 was found with them."""
        return machine()

    @app.put("/api/settings")
    def put_settings(chosen: Settings) -> dict:
        if chosen.cs2 is not None and not chosen.cs2.strip():
            chosen.cs2 = None
        if chosen.cs2 is not None and maps_folder(chosen.cs2) is None:
            raise HTTPException(
                422, "There is no CS2 there: choose the folder CS2 is installed in"
            )
        save_settings(chosen, runner.settings_file)
        runner.settings_changed()
        return machine()

    @app.get("/api/radars/{map_name}")
    def radar(map_name: str) -> FileResponse:
        """How to place world coordinates on a map's radar (render_radars.py)."""
        path = radars / f"{map_name}.json"
        if not _MAP_NAME.match(map_name) or not path.exists():
            raise HTTPException(404, "No radar for that map")
        return FileResponse(path, media_type="application/json")

    @app.get("/radars/{map_name}.png", include_in_schema=False)
    def radar_image(map_name: str) -> FileResponse:
        path = radars / f"{map_name}.png"
        if not _MAP_NAME.match(map_name) or not path.exists():
            raise HTTPException(404, "No radar for that map")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/reports/{report_id}")
    def report(report_id: str) -> FileResponse:
        path = runner.reports / f"{report_id}.json"
        if not _REPORT_ID.match(report_id) or not path.exists():
            raise HTTPException(404, "No such report")
        return FileResponse(path, media_type="application/json")

    return app


__all__ = [
    "CSP",
    "LOCAL_HOSTS",
    "MAX_DEMO_BYTES",
    "WEB",
    "create_app",
    "is_local_origin",
]
