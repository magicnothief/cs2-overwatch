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
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from overwatch import paths
from overwatch.api.jobs import Runner
from overwatch.layers.l4_judge.server import nvidia_cuda_major
from overwatch.maps.steam import find_cs2_maps, maps_folder
from overwatch.settings import Settings, save_settings

WEB = Path(__file__).resolve().parents[1] / "web"

#: Competitive demos run 50-150 MB, tournament ones up to ~500 MB.
MAX_DEMO_BYTES = 2 * 1024**3
#: Every CS2 demo starts with these bytes; CS:GO demos ("HL2DEMO") do not parse.
DEMO_MAGIC = b"PBDEMS2\x00"
_REPORT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAP_NAME = re.compile(r"^[a-z0-9_]+$")


def create_app(
    runner: Runner, *, uploads: Path = paths.UPLOADS, radars: Path = paths.RADARS
) -> FastAPI:
    app = FastAPI(title="Overwatch review", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(WEB / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        return runner.status()

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
        if head != DEMO_MAGIC:
            partial.unlink(missing_ok=True)
            raise HTTPException(422, "That is not a CS2 demo (.dem from CS2)")
        demo = uploads / f"{digest.hexdigest()[:16]}.dem"
        partial.replace(demo)
        return runner.submit(demo, Path(name).name, judge).public()

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


__all__ = ["MAX_DEMO_BYTES", "WEB", "create_app"]
