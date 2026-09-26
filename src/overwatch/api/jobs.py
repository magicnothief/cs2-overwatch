"""Analyses run one at a time, in the background, and report their progress.

One worker, not a pool: the judge holds a language model in memory (and on the
GPU when there is one), and two analyses fighting over it would each run slower
than one after the other. Models load on first use and stay loaded.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from overwatch.layers.l4_judge.judge import Judge
from overwatch.pipeline import Scorer, analyze_demo
from overwatch.settings import FILE as SETTINGS_FILE
from overwatch.settings import Settings, load_settings

log = logging.getLogger(__name__)

#: The stages a reviewer sees, in order, with the words the page uses for them.
STAGES: tuple[tuple[str, str], ...] = (
    ("parse", "Read the demo"),
    ("layer 1", "Hard limits"),
    ("layer 2", "Line of sight"),
    ("layer 3", "Behaviour scores"),
    ("layer 4", "Judge"),
)


@dataclass
class Job:
    id: str
    demo: str
    judge_policy: str
    state: str = "queued"  # queued | running | done | failed
    stage: str | None = None
    message: str = "Waiting for the previous analysis to finish"
    reached: list[str] = field(default_factory=list)
    error: str | None = None
    report_id: str | None = None
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {
            "id": self.id,
            "demo": self.demo,
            "state": self.state,
            "stage": self.stage,
            "message": self.message,
            "reached": self.reached,
            "stages": [{"key": k, "label": label} for k, label in STAGES],
            "error": self.error,
            "report_id": self.report_id,
        }


class Runner:
    """Queues analyses and runs them on a single background thread."""

    def __init__(
        self,
        reports: Path,
        *,
        scorer_path: Path,
        judge_model: Path | None,
        gpu_layers: int | str | None = None,
        settings_file: Path = SETTINGS_FILE,
    ) -> None:
        """`gpu_layers` None follows the saved setting; 0 or "auto" overrides it."""
        self.reports = reports
        self.scorer_path = scorer_path
        self.judge_model = judge_model
        self.gpu_layers = gpu_layers
        self.settings_file = settings_file
        self.jobs: dict[str, Job] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
        self._lock = threading.Lock()
        self._scorer: Scorer | None = None
        self._judge: Judge | None = None

    def settings(self) -> Settings:
        return load_settings(self.settings_file)

    def settings_changed(self) -> None:
        """A new GPU choice takes effect with the next judge: stop the running one."""
        with self._lock:
            if self._judge is not None:
                self._judge.close()
                self._judge = None

    def status(self) -> dict:
        """What the tool can do right now, for the page's first screen."""
        return {
            "scorer": self.scorer_path.exists(),
            "judge": bool(self.judge_model and self.judge_model.exists()),
            "judge_model": self.judge_model.name if self.judge_model else None,
            "busy": any(j.state in ("queued", "running") for j in self.jobs.values()),
        }

    def submit(self, demo: Path, name: str, judge_policy: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], demo=name, judge_policy=judge_policy)
        self.jobs[job.id] = job
        self._pool.submit(self._run, job, demo)
        return job

    def _models(
        self, want_judge: bool, tell: Callable[[str], None]
    ) -> tuple[Scorer, Judge | None]:
        with self._lock:
            if self._scorer is None:
                self._scorer = Scorer(self.scorer_path)
            if want_judge and self._judge is None and self.status()["judge"]:
                chosen = self.settings()
                gpu = chosen.gpu_layers if self.gpu_layers is None else self.gpu_layers
                self._judge = Judge(
                    self.judge_model,
                    gpu_layers=gpu,
                    prefer_cuda=chosen.prefer_cuda,
                    progress=tell,
                )
            return self._scorer, self._judge if want_judge else None

    def _run(self, job: Job, demo: Path) -> None:
        job.state = "running"

        def progress(stage: str, message: str) -> None:
            job.stage, job.message = stage, message
            if stage not in job.reached:
                job.reached.append(stage)

        try:
            progress("parse", "Loading the models")
            scorer, judge = self._models(
                job.judge_policy != "none", lambda m: progress("parse", m)
            )
            report = analyze_demo(
                demo,
                scorer=scorer,
                judge=judge,
                judge_policy=job.judge_policy,
                progress=progress,
                cs2=self.settings().cs2,
            )
            report.demo = job.demo
            report_id = demo.stem
            self.reports.mkdir(parents=True, exist_ok=True)
            (self.reports / f"{report_id}.json").write_text(report.model_dump_json())
            job.report_id = report_id
            job.state, job.message = "done", "Finished"
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # the page must learn what went wrong
            # BaseException too: the demo parser panics (a Rust panic, not an
            # Exception) on a truncated file, and the job must not hang "running"
            log.exception("analysis %s failed", job.id)
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "The analysis stopped"


__all__ = ["STAGES", "Job", "Runner"]
