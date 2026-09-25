# schemas: the shared data contracts

**Purpose:** Match, Moment, Evidence, Verdict (see `docs/ARCHITECTURE.md` §2). Every layer
depends on these, and nothing here depends on a layer.

**Learn first:** pydantic v2 models, type hints, JSON serialization, enums.

**First exercise (on paper, before any code):** answer these three questions.
- What does the web UI need to draw a timeline of suspicious moments?
- What does Layer 4 need to write a sentence like "reaction 94 ms vs. a legit median of 210 ms"?
- What does a reviewer need to find the moment in CS2 (`demo_gototick`)?

Your fields come from the answers.

**Done when:** a Moment with three Evidence items round-trips to JSON and back, and a test passes.

**Pitfall:** don't put per-tick arrays inside a Moment. Store tick ranges and point to the parquet.
