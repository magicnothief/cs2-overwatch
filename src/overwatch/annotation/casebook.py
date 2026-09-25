"""The cases to annotate, rendered exactly as the model sees them, in a useful order.

Order matters because nobody annotates all 1,173. Two rules:

1. **Validation cases first.** The first few hundred annotations are then a
   human-judged test set for the fine-tuned model, which is the number that says
   whether it reasons like a person. Training cases follow.
2. **Round-robin over (ban label, generated verdict).** Left alone, the queue would
   be mostly easy cases. Interleaving the four combinations puts the ones where
   generated targets are least trustworthy — banned players the evidence does not
   convict, clean players it does not clear — in front early.

The annotator never sees which stratum a case came from, the ban label, or the
generated target until they have saved their own verdict.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from functools import cached_property
from pathlib import Path

from overwatch.annotation.store import case_key
from overwatch.layers.l4_judge.rendering import PlayerCase, render_case
from overwatch.layers.l4_judge.targets import (
    build_target,
    case_rng,
    evidence_points,
    split_for,
    standard_caveats,
)
from overwatch.layers.l4_judge.verdict import Verdict


class CaseBook:
    """Read-only access to the judge cases, keyed by "match_id/player_id"."""

    def __init__(self, cases_file: str | Path, *, seed: int = 0) -> None:
        self.seed = seed
        self._rows: dict[str, dict] = {}
        for raw in Path(cases_file).read_text().splitlines():
            if raw.strip():
                row = json.loads(raw)
                self._rows[case_key(row["match_id"], row["player_id"])] = row

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, key: str) -> bool:
        return key in self._rows

    def case(self, key: str) -> PlayerCase:
        return PlayerCase.model_validate(self._rows[key]["case"])

    def evidence(self, key: str) -> str:
        """Rendered now, not read from the file: the renderer is the contract."""
        return render_case(self.case(key))

    def label(self, key: str) -> str:
        return self._rows[key]["label"]

    def split(self, key: str) -> str:
        return split_for(self._rows[key]["match_id"])

    def generated(self, key: str) -> Verdict:
        """The target the training set holds for this case when nobody annotates it."""
        case = self.case(key)
        return build_target(case, case_rng(case.match_id, case.player_id, self.seed))

    def phrases(self, key: str) -> dict[str, list[str]]:
        """Ready-made sentences for the form: every citable measurement and caveat.

        Offered as a starting point to edit, because retyping "held the crosshair on
        an enemy they could not see for 64% of the approach" two hundred times is how
        annotation sessions end early. Which to use, and whether to agree with them,
        stays the annotator's call.
        """
        case = self.case(key)
        return {
            "reasons": [text for text, _, _ in evidence_points(case)],
            "caveats": standard_caveats(case),
        }

    @cached_property
    def queue(self) -> list[str]:
        """Every case, in annotation order (see the module docstring)."""
        ordered: list[str] = []
        for half in ("val", "train"):
            strata: dict[tuple[str, str], list[str]] = defaultdict(list)
            for key in self._rows:
                if self.split(key) == half:
                    strata[(self.label(key), self.generated(key).verdict)].append(key)
            rng = random.Random(self.seed)
            groups = []
            for stratum in sorted(strata):
                keys = sorted(strata[stratum])
                rng.shuffle(keys)
                groups.append(keys)
            for batch in itertools.zip_longest(*groups):
                ordered += [key for key in batch if key is not None]
        return ordered


__all__ = ["CaseBook"]
