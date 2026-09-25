# collect: demo collector + ban labeler

**Purpose:** keep a growing, labeled pile of matchmaking demos without breaking anyone's ToS.
It has four parts:

```
share-code sources ──► downloader ──► manifest (SQLite) ──► ban checker (re-runs for weeks)
```

1. **Share-code sources**
   - Simplest, for your own account only: `csdm dl-valve` with **no share codes** downloads the
     last matchmaking demos of the Steam account logged in on this machine. It needs no auth code.
   - Your own account, plus friends who *agree* and give you their Game Authentication Code:
     Steam Web API `ICSGOPlayers_730/GetNextMatchSharingCode/v1` (steamid + steamidkey + knowncode).
   - Codes that people send you directly.
2. **Downloader:** `csdm dl-valve <sharecode...> --output data/raw/mm/`. CS Demo Manager talks to
   Steam's Game Coordinator for you; Steam must be running and logged in.
3. **Manifest** (one row per match): share code, source, match date, map, the 10 SteamIDs (read
   them with demoparser2), file path, sha256, download time.
4. **Ban checker:** Steam `GetPlayerBans`, batched. Run it at +2, +4, and +8 weeks (Valve delays
   bans on purpose) and **store every response with its timestamp**. Don't just overwrite a
   yes/no flag: `DaysSinceLastBan` only makes sense next to the date you asked.

**Learn first:** HTTP APIs with `httpx` or `requests`, environment variables and secrets, SQLite,
`subprocess`, scheduling (a systemd user timer on Arch), and idempotency (running twice must not
download twice).

**Exercises, in order:**
1. Get your own auth code and your latest share code. Call `GetNextMatchSharingCode` once with
   `curl`, and read the response.
2. Download that match with `csdm dl-valve`, then list its 10 SteamIDs with demoparser2.
3. Design the manifest table on paper, then build it.
4. Write the ban checker, and decide how a row becomes a label (ADR 0003).

**Pitfalls:**
- Valve download links **expire about 1 month after the match**, so run collection daily.
- Auth codes and API keys go in `.env`, never in git.
- Friends' auth codes expose their full match history. Ask before using them, and store them
  like passwords.
- Demos contain names and SteamIDs. Keep them local.
