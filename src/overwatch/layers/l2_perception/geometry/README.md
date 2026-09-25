# perception/geometry

**Learn first:**
- Converting pitch/yaw to a unit direction vector.
- The angle between two vectors (dot product).
- Field of view.
- Ray–triangle intersection (Möller–Trumbore).
- awpy's `VisibilityChecker` and `.tri` map files.

**Build it in steps, and validate each one before starting the next:**
1. **Baseline:** the game's `spotted` / `approximate_spotted_by` fields. They're free, but coarse.
2. **Angular distance** from the crosshair to each enemy's head. Head ≈ origin + eye height; in
   CS:GO that was ~64 units standing and ~46 crouched. Check the CS2 values yourself.
3. **Wall occlusion:** a ray from the suspect's eye to the enemy's head, tested against map triangles.
4. **Smokes:** model them as volumes between the detonate and expire events. Start with a
   sphere or cylinder.
5. **Flashes:** `player_blind` duration.

**Done when:** for any kill you can print "enemy visible at tick T, crosshair on the head at
T+k", and it matches what you see in CS2 at `demo_gototick T`.

**Pitfalls:** units (Hammer units vs. meters); the pitch sign convention; checking against the
enemy's feet instead of the head; performance (cache per tick and only query around engagements).
