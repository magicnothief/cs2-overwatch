# tests

- `pytest`. Unit-test the math first (angle wrap-around, direction vectors, angular distance).
  Bugs there silently poison everything downstream.
- `fixtures/`: a tiny trimmed parquet match (not a full demo) so tests run in milliseconds.
- **Golden set:** ~50 hand-labeled moments with expected verdicts. Run it after every change
  to any layer.
