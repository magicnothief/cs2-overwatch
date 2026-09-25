# 0007: Compute visibility by ray casting, not the game's spotting flag

- **Date:** 2026-09-22
- **Status:** accepted

## Context
Visibility ("could this attacker see the victim?") came from CS2's own spotting
system, via `approximate_spotted_by`. Validated against kills that must have had
line of sight — no wallbang, no smoke — the flag is wrong often, and wrong
*unevenly*:

| | flag says visible | ray casting |
|---|---|---|
| clean rifle kills | 82.4% | 93.7% |
| cheater rifle kills | 68.1% | 92.5% |
| clean sniper kills | 63.7% | 86.7% |
| cheater sniper kills | 36.2% | 84.0% |

A correct instrument cannot depend on who is holding what. The flag's 14-27 point
gaps between labels were measurement bias; ray casting's are 1-3 points.

## Decision
Visibility is computed by casting a ray from the shooter's eye to the target's head
against the map's collision mesh, extracted from an installed CS2 with
`training/rules/extract_maps.py`. Meshes live in `data/maps/*.tri`; matches on maps
without one fall back to the flag, and `visibility_source` records which answered.

Two implementation notes that cost real time:
- awpy's ray casting does 75 rays/second (47 hours for this dataset). Embree via
  trimesh does ~27,000, so that is what runs.
- The physics mesh contains clip volumes and the skybox, which stop players but not
  sight. Leaving them in costs nine points of accuracy.

Each extracted mesh is accepted only if it agrees with ≥82% of guaranteed-line-of-
sight kills on that map, **and** blocks ≥50% of random pairs of opposing players
(see the correction below). The current nine agree with 84-93% of kills and block
98.7-99.7% of random pairs.

## Consequences
- Wall-aim now separates within every weapon class: rifles 0.21 vs 0.08, snipers
  0.48 vs 0.27, other 0.29 vs 0.09.
- It did **not** fix the weapon confound. Clean players' scores still correlate with
  their sniper share at r=0.63 (was 0.66), because cheaters in CS2CD simply take 60%
  of kills with snipers against 15% for clean players. That is a property of the
  dataset, not of the instrument, and needs a separate fix (see 0008).
- Accuracy barely moved: per-player out-of-fold 0.938 -> 0.931. The gain here is
  validity, not score.

## Correction (2026-09-23)
The acceptance test above originally had only its first half, and it let two broken
meshes through. de_overpass was mirrored on x and de_vertigo had its horizontal
axes swapped and mirrored, so each sat beside the playable area instead of around
it. Nothing blocked any ray; the kills test only asks "is a visible kill visible?",
so an empty world scored 100% and 98%. Every visibility feature on those two maps
read "visible" (enemy visible 1.5 s before the kill: 100% and 94% of the time,
against 15-18% on every other map).

Found by the new "last seen" context, which was never null on overpass. Fixed by
the second test: random opposing pairs, nearly all of which have a wall between
them on a real map. `extract_maps.py --retest` re-checks existing meshes without the
game installed. After the fix all nine maps use the same axis convention, which is
itself a sign it is right: one exporter, one convention.

Effect on the detector: none measurable. Per-player out-of-fold AUC 0.931 -> 0.932
(90% CI 0.92-0.94); weapon confound r=0.627 -> 0.640, inside its noise. (The first
run after the fix read a stale sequences.npz — no script rebuilt it — and happened
to print the same AUC. The figures above are from the rebuilt arrays;
training/scorer/build_sequences.py now exists so that cannot recur.) Those two maps are 12% of kills, and the models
lean on aim shape more than visibility. The cost was to the judge's evidence text,
which would have told it that nobody on overpass ever aimed through a wall.

## More maps (2026-09-25)
Meshes now exist for 23 maps. cs_office and cs_italy were validated against CS2CD
(87% and 91% of sighted kills visible, 99.7% of random pairs blocked) and
de_eldorado against a local demo; nine maps nobody has recorded were written with
the axis convention all validated maps share and are marked unvalidated in
meshes.json, to be checked when a demo from one of them arrives.

The two CS2CD maps that switched from the spotting flag to ray casting cover
6,159 kills. After rebuilding the dataset and retraining (label `meshes23`), the
detector is unchanged within noise: per-player out-of-fold AUC 0.932 (0.92-0.94),
LightGBM 0.918. As with the first nine maps, the gain is validity: on a Wingman
demo the flag had made one player look as if he aimed through walls, and real
line of sight showed the enemies were in view.
