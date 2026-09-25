# Layer 1: blatant rules

**Purpose:** cheap, high-precision rules for behavior that is physically or game-mechanically
impossible. See the rule table in `docs/ARCHITECTURE.md` §4.

**Learn first:** distributions, percentiles, and why a threshold should come from data rather
than intuition.

**First exercise:**
1. Compute each player's maximum sustained yaw speed across ~20 pro demos and plot the distribution.
2. Choose a threshold and write down why you chose it.
3. Run it on the CS2CD with-cheater set. How many flagged players are labeled cheaters?

**Interface idea:** each rule is a function `(match, perception) -> list[Evidence]`. Rules
shouldn't know about each other.

**Done when:** zero false positives on pro demos, every hit carries value + threshold + note,
and you've watched 10 flags yourself in CS2.

**Pitfalls:** round-start teleports, respawns in deathmatch, spectator/death-cam angles,
angle wrap-around (again).
