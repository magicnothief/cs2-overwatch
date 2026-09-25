"""Run the judge model, on a GPU when one has room for it, else on the CPU.

Sized for a mid-range desktop: a 4B model at Q4_K_M is about 2.7 GB of RAM and
needs no GPU. Generation is the slow part, so the prompt is short, the output is
a small JSON object, and the context window is kept modest.

The model runs in llama.cpp's own prebuilt server (server.py picks and downloads
the build for this machine); this class renders the prompt and reads the answer.
The same class serves the stock model and the fine-tuned one — only the GGUF path
changes — so a fine-tune from Unsloth Studio drops in without touching callers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from overwatch import models, paths
from overwatch.layers.l4_judge import server
from overwatch.layers.l4_judge.rendering import SYSTEM_PROMPT, PlayerCase, render_case
from overwatch.layers.l4_judge.verdict import VERDICT_SCHEMA, Verdict

#: The judge in use: fine-tune v3 (evidence-only targets, clean-player lines shown
#: in the text), exported from Unsloth Studio. Earlier versions and how they
#: compare: training/llm/README.md.
DEFAULT_MODEL = models.JUDGE.local
#: The untuned model every fine-tune is measured against (baseline_judge.py).
STOCK_MODEL = paths.MODELS / "llm" / "qwen3.5-4b-instruct-Q4_K_M.gguf"
#: Where the downloaded llama.cpp builds live.
DEFAULT_ENGINES = paths.ENGINES


@dataclass
class JudgeResult:
    """A verdict plus what it cost to produce, since CPU time is the budget here.

    `verdict` is None when the model ran out of tokens mid-answer. The grammar
    guarantees the *shape* of the JSON, not that it finishes, so truncation is a
    real failure mode on a small model and worth counting rather than crashing on.
    """

    verdict: Verdict | None
    raw: str
    seconds: float
    tokens: int
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.verdict is not None

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


class Judge:
    """A local language model that turns evidence into a written verdict."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        *,
        context: int = 4096,
        threads: int | None = None,
        gpu_layers: int | str = "auto",
        seed: int = 0,
        engines: Path = DEFAULT_ENGINES,
        prefer_cuda: bool = False,
        progress: server.Progress | None = None,
    ) -> None:
        """Args:
        model_path: the GGUF to load.
        context: prompt + answer budget, in tokens. Cases run to ~1,700 tokens
            and answers may take 400: 2,048 was too small for the longest.
        threads: CPU threads; llama.cpp picks a sensible default when None.
        gpu_layers: "auto" uses a GPU when one has room for the model, 0 keeps
            the judge on the CPU (and off every GPU). See server.py.
        seed: fixed so a report can be reproduced.
        engines: where llama.cpp builds are downloaded to and kept.
        prefer_cuda: on NVIDIA, try the CUDA build before Vulkan.
        progress: told what is happening while an engine downloads or starts.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            msg = f"no GGUF at {self.model_path}"
            raise FileNotFoundError(msg)
        self.seed = seed
        self.engine = server.start(
            self.model_path,
            engines=engines,
            gpu_layers=gpu_layers,
            context=context,
            threads=threads,
            prefer_cuda=prefer_cuda,
            progress=progress,
        )

    def describe(self) -> str:
        """Where the judge runs, for reports: "GPU: NVIDIA ... via cuda-13 (...)"."""
        return self.engine.describe()

    def close(self) -> None:
        """Stop the engine; the judge cannot be used afterwards."""
        self.engine.close()

    def __del__(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.close()

    def judge(self, case: PlayerCase, *, max_tokens: int = 400) -> JudgeResult:
        """Read one player's evidence and return a structured verdict."""
        return self.judge_text(render_case(case), max_tokens=max_tokens)

    def judge_text(self, evidence: str, *, max_tokens: int = 400) -> JudgeResult:
        started = time.perf_counter()
        response = self.engine.post(
            "/completion",
            {
                "prompt": server.render_prompt(SYSTEM_PROMPT, evidence),
                # the grammar makes invalid JSON impossible rather than unlikely
                "json_schema": VERDICT_SCHEMA,
                "temperature": 0.0,
                "seed": self.seed,
                "n_predict": max_tokens,
                "cache_prompt": False,  # the same case gives the same answer
            },
        )
        elapsed = time.perf_counter() - started
        raw = response["content"]
        tokens = response["tokens_predicted"]

        try:
            verdict = Verdict.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            return JudgeResult(None, raw, elapsed, tokens, error=str(exc)[:200])
        return JudgeResult(verdict, raw, elapsed, tokens)
