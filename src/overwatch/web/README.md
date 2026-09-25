# web: frontend

**Pages:** upload → jobs → match report (players ranked by score) → moment detail.

**Moment detail** is the heart of the UI:
- A 2D radar replay of the moment (map image plus player dots; awpy has map coordinate transforms).
- Aim-angle and angular-speed charts with the kill and first-sight ticks marked.
- An evidence table (value vs. legit baseline).
- The verdict and reasoning.
- The command to jump there in CS2 (`demo_gototick <tick>`).

**Stack choice** (record it as an ADR):
- **HTMX + Jinja templates (recommended first):** almost no JavaScript, so you stay focused on
  the ML.
- **Svelte or React:** more to learn. Worth it later if the UI grows.

**Done when:** a friend can review a match without you explaining anything.
