# web: the review page

**Pages:** drop a demo → the analysis running → the match report → one moment.

**What it is built from:** `index.html`, `app.css`, `app.js`. No framework, no
build step, one variable font. The server hands `app.js` a report as JSON and it
renders the markup. Why it looks the way it does — tokens, spacing, type ramp,
the order of the report, keyboard and screen-reader rules — is in `DESIGN.md`,
beside this file. Read that before changing a colour or a spacing.

**The report, in order:** the answer (who is flagged, and the verdict) → the
timeline of every kill → the moments to watch → the moment itself, with radar,
crosshair trace, what was measured, and the `demo_gototick` command that takes
you there in CS2 → the player panel with the judge's reasons and caveats.

**Looking at it without a demo:**

```
python -m overwatch.web.fake_report    # two synthetic reports + their URLs
python -m overwatch.api                # then open the URL it printed
```

`fake_report.py` invents every name, Steam ID, score and tick, and writes both a
report with a flagged player and one where nobody is flagged. Screenshots for
issues and docs come from it: no real player's name or Steam ID ever leaves this
machine.

**Done when:** a friend can review a match without you explaining anything.
