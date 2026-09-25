# Layer 2: perception ("what could the suspect see?")

**Purpose:** a *service* that Layers 1 and 3 call. It is not a pipeline stage. Two backends
share one interface:
- `geometry/`: default, CPU, exact, fast.
- `vision/`: optional; YOLO on rendered clips.

Interface idea: `visible_enemies(match, steamid, tick) -> list[Sighting]`, where a Sighting holds
enemy, on_screen, occluded, in_smoke, suspect_flashed, and angular_distance_to_crosshair.

Read `docs/ARCHITECTURE.md` §3.1 for why geometry is the default.
