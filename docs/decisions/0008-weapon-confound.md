# 0008: The weapon confound is in the data, and invariance is not free

- **Date:** 2026-09-22
- **Status:** accepted

## Context
Cheaters in CS2CD take 60% of their kills with snipers; clean players take 15%.
Sniping is kinematically distinctive, so the scorer learns it, and clean AWP mains
pay for it: their scores correlate with their sniper share at r=0.64, and the clean
players flagged at the 95th percentile have a median sniper share of 0.83.

Two fixes were tried.

**Fixing the instrument (ADR 0007).** The game's spotting flag was badly biased by
weapon; ray casting removed that bias almost entirely. It did not move the
confound: r fell only 0.66 -> 0.63.

**Adversarial invariance.** A weapon-classifier head behind a gradient-reversal
layer, so the encoder is punished for keeping weapon information. Five-fold,
out-of-fold, same protocol throughout:

| strength | ROC-AUC | PR-AUC | r(clean, sniper) | AUC, AWP mains | AUC, rifle players |
|---|---|---|---|---|---|
| 0.0 | 0.905 | 0.711 | 0.639 | 0.718 | 0.805 |
| 0.5 | 0.898 | 0.705 | 0.596 | 0.717 | 0.789 |
| 1.5 | 0.863 | 0.663 | **0.443** | 0.686 | 0.732 |

## Decision
Keep the adversary available, default it to 0.5, and never report a headline number
without the per-weapon-group numbers beside it.

The strength-1.5 result is what settles this. If the adversary were only deleting a
shortcut, accuracy *within* a weapon group would hold — the confound cannot help
when every player in the group uses the same weapon. Instead, rifle-player accuracy
falls from 0.805 to 0.732. The adversary is removing real behaviour, not just the
proxy, because how a cheat shows up genuinely differs between an AWP and an AK.

## Consequences
- Reported accuracy drops, honestly: the plain model's 0.905 includes a component
  that would not transfer to a player pool with a different weapon mix.
- The tool over-flags AWP mains. Until this is fixed, that belongs in the report
  the reviewer reads, not only in a decision record.
- The real fix is data: matchmaking demos with ban labels, where the weapon mix of
  cheaters reflects the population being judged rather than CS2CD's sample. That is
  the collector in `src/overwatch/collect/`.
