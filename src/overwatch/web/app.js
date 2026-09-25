// Overwatch review: three views, routed by the URL hash.
//   #/                      drop a demo, or open an earlier review
//   #/job/<id>              an analysis in progress
//   #/report/<id>?p=&k=     a report, with a player (p) and a kill tick (k) selected

const view = document.getElementById("view");
const barAction = document.getElementById("bar-action");

const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style") node.setAttribute("style", value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
};

async function api(path) {
  const res = await fetch(path);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
  return body;
}

/** Weapon ids as the game's buy menu names them. */
const WEAPONS = {
  ak47: "AK-47", m4a1: "M4A4", m4a1_silencer: "M4A1-S", m4a4: "M4A4", awp: "AWP", ssg08: "SSG 08",
  scar20: "SCAR-20", g3sg1: "G3SG1", aug: "AUG", sg556: "SG 553", galilar: "Galil AR", famas: "FAMAS",
  deagle: "Desert Eagle", revolver: "R8 Revolver", glock: "Glock-18", hkp2000: "P2000",
  usp_silencer: "USP-S", p250: "P250", fiveseven: "Five-SeveN", tec9: "Tec-9", cz75a: "CZ75-Auto",
  elite: "Dual Berettas", mac10: "MAC-10", mp9: "MP9", mp7: "MP7", mp5sd: "MP5-SD", ump45: "UMP-45",
  p90: "P90", bizon: "PP-Bizon", nova: "Nova", xm1014: "XM1014", mag7: "MAG-7", sawedoff: "Sawed-Off",
  m249: "M249", negev: "Negev", knife: "knife", taser: "Zeus x27", hegrenade: "HE grenade",
  inferno: "molotov", molotov: "molotov", incgrenade: "incendiary", world: "the world",
};
const weapon = (id) => (id ? WEAPONS[id] || (id.startsWith("knife") || id.startsWith("bayonet") ? "knife" : id) : null);

/** A button that copies `text`, and says whether that worked. */
function copyButton(text) {
  const button = el("button", { type: "button", class: "quiet" }, "Copy");
  button.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(text); button.textContent = "Copied"; }
    catch { button.textContent = "Select and copy it"; }
    setTimeout(() => (button.textContent = "Copy"), 1600);
  });
  return button;
}

const sentence = (text) => (text ? text[0].toUpperCase() + text.slice(1) : "");
const pct = (share) => `${Math.round(100 * share)}%`;
const timeAgo = (seconds) => {
  const date = new Date(seconds * 1000);
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
};

/* ================================================================== home */

async function home() {
  barAction.hidden = true;
  const status = await api("/api/status").catch(() => null);
  const reports = await api("/api/reports").catch(() => []);
  const machine = await api("/api/settings").catch(() => null);

  const input = el("input", { type: "file", accept: ".dem", hidden: true });
  const judgeChoice = (value, label, checked) =>
    el("label", {}, el("input", { type: "radio", name: "judge", value, checked }), " ", label);

  const zone = el(
    "section",
    { class: "drop", "aria-label": "Analyse a demo" },
    el("h1", {}, "Drop a CS2 demo to review it"),
    el("br"),
    el(
      "p",
      { class: "lede" },
      "It stays on this computer. Scores take about ten seconds; the judge adds half a minute for each player it reads.",
    ),
    el(
      "div",
      { class: "actions" },
      el("button", { type: "button", onclick: () => input.click() }, "Choose a demo"),
      el(
        "fieldset",
        {},
        el("legend", {}, "The judge reads"),
        judgeChoice("flagged", "flagged players", true),
        judgeChoice("all", "everyone"),
        judgeChoice("none", "nobody"),
      ),
    ),
    !status
      ? el("p", { class: "warning" }, "The app is not answering. Start it again with: overwatch")
      : !status.scorer
        ? el("p", { class: "warning" }, "The detector model is missing. Run overwatch setup, then reload.")
        : !status.judge
          ? el("p", { class: "warning" }, "No judge model was found, so reports will have scores but no written verdicts.")
          : null,
    input,
  );

  const start = (file) => {
    if (!file) return;
    const judge = zone.querySelector("input[name=judge]:checked").value;
    upload(file, judge);
  };
  input.addEventListener("change", () => start(input.files[0]));
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    start(e.dataTransfer.files[0]);
  });

  const history = reports.length
    ? el(
        "table",
        { class: "history" },
        el("thead", {}, el("tr", {},
          el("th", {}, "Map"), el("th", {}, "Demo"), el("th", { class: "num" }, "Kills"),
          el("th", {}, "Flagged"), el("th", {}, "Reviewed"))),
        el("tbody", {}, reports.map((r) =>
          el("tr", { "data-id": r.id, tabindex: 0,
              onclick: () => (location.hash = `#/report/${r.id}`),
              onkeydown: (e) => e.key === "Enter" && (location.hash = `#/report/${r.id}`) },
            el("td", { class: "map" }, (r.map_name || "unknown").replace(/^de_|^cs_/, "")),
            el("td", {}, r.demo),
            el("td", { class: "num" }, r.kills),
            el("td", { class: r.flagged ? "flag" : "muted" },
              r.flagged ? `${r.flagged} of ${r.players}` : "nobody"),
            el("td", { class: "muted" }, timeAgo(r.analysed)),
          ))),
      )
    : el("p", { class: "muted" }, "Reviews you run appear here.");

  view.replaceChildren(status?.update ? updateNotice(status) : "", zone,
    el("h2", {}, "Earlier reviews"), el("div", { style: "margin-top:1rem" }, history),
    machine ? settingsPanel(machine) : "");
}

/** A newer release is out: what it is, where its notes are, how to install it. */
function updateNotice(status) {
  const u = status.update;
  return el("aside", { class: "update", "aria-label": "Update available" },
    el("p", {}, el("strong", {}, `Version ${u.version} is out`), ` (this is ${status.version}). `,
      u.url ? el("a", { href: u.url, target: "_blank", rel: "noopener" }, "What's new") : null,
      u.url ? ". " : "", "To update, close the app and run:"),
    el("div", { class: "goto" }, el("code", {}, u.command), copyButton(u.command)));
}

/** Where CS2 is, and whether the judge may use the GPU: saved on this computer. */
function settingsPanel(machine) {
  const said = el("p", { class: "said", role: "status" });
  const save = async (change) => {
    const res = await fetch("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cs2: machine.cs2, gpu: machine.gpu, updates: machine.updates !== false, ...change }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) { said.textContent = body.detail || "That was not saved."; said.className = "said flag"; return; }
    machine = body;
    said.textContent = "Saved."; said.className = "said muted";
    draw();
  };

  const folder = el("input", { type: "text", spellcheck: "false", autocomplete: "off",
    placeholder: "The folder CS2 is installed in", "aria-label": "CS2 folder" });
  const form = el("form", { class: "cs2-form" }, folder,
    el("button", { type: "submit", class: "quiet" }, "Save"));
  form.addEventListener("submit", (e) => { e.preventDefault(); save({ cs2: folder.value.trim() || null }); });

  const gpu = (value, label) => el("label", {},
    el("input", { type: "radio", name: "gpu", value, checked: machine.gpu === value,
      onchange: () => save({ gpu: value }) }), " ", label);

  const cs2Row = el("dd", {});
  const draw = () => {
    folder.value = machine.cs2 || "";
    cs2Row.replaceChildren(
      machine.cs2_found
        ? el("p", {}, "Found. Each map is read from it once, the first time a demo needs it: ",
            el("code", {}, machine.cs2_found))
        : el("p", {}, el("span", { class: "flag" }, "Not found. "),
            "Without it the line of sight falls back to the game's own spotting, which is less reliable. ",
            "Steam lists where it installed CS2; a library on another drive or system may not be listed."),
      form,
      el("p", { class: "muted hint" }, "Leave it empty to find CS2 through Steam."),
    );
  };
  draw();

  return el("section", { class: "machine", "aria-labelledby": "machine-title" },
    el("h2", { id: "machine-title" }, "This computer"),
    el("dl", {},
      el("dt", {}, "CS2"), cs2Row,
      el("dt", {}, "Judge"),
      el("dd", {}, el("fieldset", {}, el("legend", { class: "sr-only" }, "Where the judge runs"),
        gpu("auto", "on the graphics card when it has room"),
        machine.nvidia || machine.gpu === "cuda"
          ? gpu("cuda", "on the graphics card through CUDA: a little faster, a 600 MB download once")
          : null,
        gpu("off", "on the processor only, leaving the graphics card free"))),
      el("dt", {}, "Updates"),
      el("dd", {}, el("label", {},
        el("input", { type: "checkbox", checked: machine.updates !== false,
          onchange: (e) => save({ updates: e.target.checked }) }),
        " Ask GitHub once a day whether a new version is out")),
      el("dt", {}, "Files"),
      el("dd", {}, el("p", {}, "Models, maps and reviews are kept in ", el("code", {}, machine.home)))),
    said);
}

function upload(file, judge) {
  const stages = el("ol", { class: "stages" }, el("li", { class: "now" }, "Upload", el("span", { class: "message" }, "Starting")));
  view.replaceChildren(el("section", { class: "job" }, el("h1", {}, file.name), stages));
  const message = stages.querySelector(".message");

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/analyses?name=${encodeURIComponent(file.name)}&judge=${judge}`);
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) message.textContent = `${Math.round((100 * e.loaded) / e.total)}% of ${(e.total / 1e6).toFixed(0)} MB`;
  };
  xhr.onload = () => {
    const body = JSON.parse(xhr.responseText || "{}");
    if (xhr.status >= 400) return failed(file.name, body.detail || xhr.statusText);
    location.hash = `#/job/${body.id}`;
  };
  xhr.onerror = () => failed(file.name, "The upload did not reach the server.");
  xhr.send(file);
}

function failed(title, why) {
  view.replaceChildren(
    el("section", { class: "job" },
      el("h1", {}, title),
      el("p", { class: "warning" }, why),
      el("p", {}, el("a", { href: "#/" }, "Choose another demo"))),
  );
}

/* ================================================================ a job */

let polling = null;

async function job(id) {
  barAction.hidden = true;
  const draw = (j) => {
    const stages = el("ol", { class: "stages" },
      j.stages.map((s) => {
        const now = j.stage === s.key && j.state === "running";
        const done = j.reached.includes(s.key) && !now;
        return el("li", { class: now ? "now" : done ? "done" : "" }, s.label,
          now ? el("span", { class: "message" }, j.message) : null);
      }));
    view.replaceChildren(el("section", { class: "job" },
      el("h1", {}, j.demo),
      j.state === "queued" ? el("p", { class: "muted" }, j.message) : null,
      stages));
  };
  const tick = async () => {
    let j;
    try { j = await api(`/api/analyses/${id}`); } catch (e) { return failed("Analysis", e.message); }
    if (j.state === "done") return (location.hash = `#/report/${j.report_id}`);
    if (j.state === "failed") return failed(j.demo, j.error || "The analysis stopped.");
    draw(j);
    polling = setTimeout(tick, 700);
  };
  tick();
}

/* ============================================================= a report */

const reportCache = new Map();
const radarCache = new Map();

/** Where a map's radar sits in world coordinates, or null when there is none. */
async function radarFor(map) {
  if (!map) return null;
  if (!radarCache.has(map)) radarCache.set(map, await api(`/api/radars/${map}`).catch(() => null));
  return radarCache.get(map);
}

async function report(id, params) {
  barAction.hidden = false;
  let data = reportCache.get(id);
  if (!data) {
    try { data = await api(`/api/reports/${id}`); } catch (e) { return failed("Report", e.message); }
    reportCache.set(id, data);
  }
  const players = orderedPlayers(data);
  const highest = [...players].sort((a, b) => (b.flagged - a.flagged) || ((b.score ?? -1) - (a.score ?? -1)))[0];
  const player = players.find((p) => p.player_id === params.get("p")) || highest;
  const kills = player ? player.kill_log : [];
  const kill = kills.find((k) => String(k.tick) === params.get("k"))
    || (player && player.top_kills[0]) || kills[0];

  const radar = await radarFor(data.map_name);
  const fresh = view.dataset.report !== id;
  view.dataset.report = id;
  const detail = el("section", { class: "detail" },
    el("div", {}, kill ? killDetail(kill, radar, data.map_name, player.name || player.player_id) : el("p", { class: "muted" }, "This player has no kills to show.")),
    player ? playerPanel(player, data) : null);
  // choosing another kill in the same report keeps the timeline as it is: rebuilt,
  // every ring would be laid out again and the lanes' sideways scroll would reset
  const kept = !fresh && view.querySelector(".timeline");
  if (kept) {
    markSelection(kept, player, kill);
    view.querySelector(".detail").replaceWith(detail);
  } else {
    view.replaceChildren(matchHeader(data), timeline(data, players, player, kill, fresh), detail);
    dodgeKills();
  }
  keyboard = (e) => navigate(e, id, players, player, kill);
}

/** Move the timeline's highlight to this player and kill without redrawing it. */
function markSelection(grid, player, kill) {
  const id = player ? player.player_id : null;
  grid.querySelectorAll("[data-player]").forEach((n) => n.classList.toggle("selected", n.dataset.player === id));
  grid.querySelectorAll(".kill.selected").forEach((n) => n.classList.remove("selected"));
  if (id && kill) {
    grid.querySelector(`.cell[data-player="${CSS.escape(id)}"] .kill[data-tick="${kill.tick}"]`)?.classList.add("selected");
  }
}

/** Teams together (by starting side), flagged players and high scores first. */
function orderedPlayers(data) {
  const sideOrder = { CT: 0, T: 1 };
  return [...data.players].sort((a, b) =>
    (sideOrder[a.side] ?? 2) - (sideOrder[b.side] ?? 2)
    || (b.flagged - a.flagged)
    || ((b.score ?? -1) - (a.score ?? -1)));
}

function matchHeader(data) {
  const map = (data.map_name || "Unknown map").replace(/^de_|^cs_/, "");
  return el("section", { class: "match" },
    el("h1", {}, map),
    el("p", { class: "summary" },
      `${data.demo}, ${data.rounds.length} rounds, ${data.kills} kills. `,
      data.visibility.startsWith("ray cast")
        ? "Who could see whom was ray cast against the map itself."
        : "Who could see whom comes from the game's own spotting flag, which is less reliable."),
    data.notes.length ? el("ul", { class: "notes" }, data.notes.map((n) => el("li", {}, sentence(n)))) : null);
}

/**
 * Kills close together in one round would merge into a blob. Each ring moves the
 * least it can, up or down inside the lane first and then sideways by up to a
 * ring's width, until it touches no ring already placed. Across a two-minute
 * round a few pixels is a fraction of a second, so the timeline still reads true.
 * The biggest (most suspicious) rings are placed first and keep their spot.
 */
function dodgeKills() {
  const ROOM = 7; // px a ring may move off the lane's centre line
  const EDGE = 3; // px kept clear of the round's border line, halo included
  for (const cell of view.querySelectorAll(".timeline .cell")) {
    const marks = [...cell.querySelectorAll(".kill")];
    const width = cell.getBoundingClientRect().width;
    // every position a ring can take stays inside its own round, so a kill at the
    // very end of one round never touches one at the start of the next
    const inside = (x, r) => Math.min(Math.max(r + EDGE, width - r - EDGE), Math.max(r + EDGE, x));
    const placed = [];
    const rings = marks
      .map((m) => ({ m, x: Number(m.dataset.at) * width, r: parseFloat(m.style.getPropertyValue("--d")) / 2 }))
      .sort((a, b) => b.r - a.r);
    for (const ring of rings) {
      const home = inside(ring.x, ring.r);
      const moves = [];
      for (const dy of [0, -4, 4, -ROOM, ROOM]) {
        for (const dx of [0, ring.r, -ring.r, 2 * ring.r, -2 * ring.r]) {
          moves.push({ x: inside(home + dx, ring.r), dy });
        }
      }
      moves.sort((a, b) => Math.abs(a.x - home) + Math.abs(a.dy) - (Math.abs(b.x - home) + Math.abs(b.dy)));
      let best = { x: home, dy: 0, overlap: Infinity };
      for (const { x, dy } of moves) {
        const overlap = placed.reduce((sum, p) =>
          sum + Math.max(0, ring.r + p.r + 1.5 - Math.hypot(x - p.x, dy - p.y)), 0);
        if (overlap < best.overlap) best = { x, dy, overlap };
        if (overlap === 0) break;
      }
      placed.push({ r: ring.r, x: best.x, y: best.dy });
      ring.m.style.setProperty("--dx", `${best.x - ring.x}px`);
      ring.m.style.setProperty("--dy", `${best.dy}px`);
    }
  }
}
new ResizeObserver(() => view.querySelector(".timeline") && dodgeKills()).observe(view);

function timeline(data, players, selected, kill, animate) {
  const rounds = data.rounds;
  const switches = new Set(data.side_switches || []);
  const grid = el("div", {
    class: "timeline",
    role: "grid",
    "aria-label": "Every kill, by player and round",
    style: `grid-template-columns: minmax(8rem, 13rem) repeat(${rounds.length}, minmax(2rem, 1fr)) minmax(12rem, auto)`,
  });

  grid.append(el("div", { class: "head name" }, "Player"));
  rounds.forEach((r) => grid.append(el("div", { class: "head" + (switches.has(r.number - 1) ? " switch" : "") }, r.number)));
  grid.append(el("div", { class: "head score" }, "Behaviour score"));

  let lastSide = null;
  for (const p of players) {
    if (lastSide !== null && p.side !== lastSide) grid.append(el("div", { class: "gap", "aria-hidden": "true" }));
    lastSide = p.side;
    const isSelected = selected && p.player_id === selected.player_id;
    const watch = new Set(p.flagged ? p.top_kills.map((k) => k.tick) : []);
    const select = () => go({ p: p.player_id, k: (p.top_kills[0] || p.kill_log[0] || {}).tick });

    grid.append(el("button", {
      class: "lane-name" + (p.flagged ? " flagged" : "") + (isSelected ? " selected" : ""),
      "data-player": p.player_id,
      style: `--side: var(--${p.side === "CT" ? "ct" : p.side === "T" ? "t" : "grid"})`,
      title: `${p.name || p.player_id}, started as ${p.side || "unknown side"}`,
      onclick: select,
    }, p.name || p.player_id));

    const byRound = new Map();
    for (const k of p.kill_log) {
      if (!byRound.has(k.round)) byRound.set(k.round, []);
      byRound.get(k.round).push(k);
    }
    rounds.forEach((r, i) => {
      const cell = el("div", {
        class: "cell" + (switches.has(r.number - 1) ? " switch" : "") + (isSelected ? " selected" : ""),
        "data-player": p.player_id,
      });
      for (const k of byRound.get(r.number) || []) {
        const at = (k.tick - r.start_tick) / Math.max(1, r.end_tick - r.start_tick);
        const size = 9 + Math.round(9 * (k.score ?? 0));
        const label = `Round ${r.number}, ${weapon(k.weapon) || "kill"}${k.headshot ? " headshot" : ""}` +
          `${k.victim ? ` on ${k.victim}` : ""}, kill score ${(k.score ?? 0).toFixed(2)}`;
        cell.append(el("button", {
          class: "kill" + (k.headshot ? " headshot" : "") + (watch.has(k.tick) ? " watch" : "") +
            (kill && isSelected && k.tick === kill.tick ? " selected" : ""),
          "data-at": ((4 + 92 * at) / 100).toFixed(4),
          "data-tick": k.tick,
          style: `left:${(4 + 92 * at).toFixed(1)}%; --d:${size}px; --round:${animate ? i : 0};` +
            (animate ? "" : "animation:none"),
          title: label,
          "aria-label": label,
          onclick: () => go({ p: p.player_id, k: k.tick }),
        }));
      }
      grid.append(cell);
    });

    const pctText = p.clean_percentile == null ? "" : `above ${pct(p.clean_percentile)} of clean players`;
    grid.append(el("div", { class: "lane-score" + (isSelected ? " selected" : ""), "data-player": p.player_id },
      el("span", { class: "value" }, p.score == null ? "–" : p.score.toFixed(2)),
      el("span", { class: "muted" }, p.enough_kills ? pctText : "too few kills to say"),
      p.verdict ? el("span", { class: `verdict ${p.verdict.verdict}` }, `${sentence(p.verdict.verdict)} ${p.verdict.probability}%`) : null));
  }

  // one row's cells light up together when the pointer is on any of them
  grid.addEventListener("pointerover", (e) => {
    const row = e.target.closest("[data-player]");
    grid.querySelectorAll(".row-hover").forEach((n) => n.classList.remove("row-hover"));
    if (row) grid.querySelectorAll(`[data-player="${CSS.escape(row.dataset.player)}"]`).forEach((n) => n.classList.add("row-hover"));
  });
  grid.addEventListener("pointerleave", () => grid.querySelectorAll(".row-hover").forEach((n) => n.classList.remove("row-hover")));

  const switchText = data.side_switches && data.side_switches.length
    ? ` The double line is where the teams swapped sides.` : "";
  return el("section", {},
    el("div", { class: "timeline-scroll" }, grid),
    el("p", { class: "legend" },
      "Each ring is a kill, placed where it happened in the round. Filled means a headshot; larger means the model found that kill more suspicious. ",
      el("span", { class: "flag" }, "Magenta"),
      " marks flagged players and the kills to watch first. The bar by each name is the side they started on: ",
      el("span", { style: "color:var(--ct);font-weight:600" }, "CT"), " or ",
      el("span", { style: "color:var(--t);font-weight:600" }, "T"), ".", switchText));
}

function killDetail(k, radarMeta, mapName, shooter) {
  const facts = [];
  const what = [weapon(k.weapon), k.headshot ? "headshot" : null].filter(Boolean).join(" ");
  if (k.distance != null) facts.push(`${Math.round(k.distance)} m away${k.walls_penetrated ? `, through ${k.walls_penetrated === 1 ? "a wall" : `${k.walls_penetrated} walls`}` : ""}.`);
  if (k.aim_through_cover != null) facts.push(`The crosshair was on the enemy through cover for ${pct(k.aim_through_cover)} of the last half second.`);
  if (k.seen_first === false) facts.push("The enemy was never visible before dying.");
  else if (k.reaction_ms != null && k.reaction_ms < 1990) facts.push(`The kill came ${Math.round(k.reaction_ms)} ms after the enemy became visible.`);
  if (k.arrival_delay_ms != null) {
    facts.push(k.arrival_delay_ms === 0
      ? "The first shot came on the very tick the crosshair reached the head."
      : `The first shot came ${Math.round(k.arrival_delay_ms)} ms after the crosshair reached the head.`);
  }
  if (k.context) facts.push(`${sentence(k.context)}.`);
  if (k.score != null) facts.push(`The model scores this kill ${k.score.toFixed(2)} out of 1.`);

  const copy = copyButton(k.demo_command);

  const chart = trace(k.trajectory, k.shots_ms || []);
  const map = radarView(k, radarMeta, mapName, shooter);
  const moments = k.path.map((p) => p.ms);
  const shot = Math.max(0, moments.indexOf(0));
  const show = (i) => { map.show(i); chart.cursor?.(moments[i]); };
  chart.onScrub?.((ms) => {
    const i = moments.reduce((best, m, j) => (Math.abs(m - ms) < Math.abs(moments[best] - ms) ? j : best), 0);
    show(i);
  }, () => show(Number(map.slider.value)));
  map.slider.addEventListener("input", () => show(Number(map.slider.value)));
  map.slider.value = shot;
  show(shot);

  return el("div", {},
    el("h2", {}, `Round ${k.round ?? "?"}: ${what || "kill"}${k.victim ? ` on ${k.victim}` : ""}`),
    el("div", { class: "visuals" },
      map.figure,
      el("div", {},
        chart,
        el("div", { class: "trace-key" },
          el("span", {}, el("span", { class: "swatch", style: "background:var(--paper)" }), "enemy visible"),
          el("span", {}, el("span", { class: "swatch", style: "background:repeating-linear-gradient(45deg,#f2f4f6 0 3px,#c9d0d7 3px 5px)" }), "enemy behind cover"),
          (k.shots_ms || []).length ? el("span", {}, el("span", { class: "swatch fire" }), "a shot") : null),
        el("p", { class: "legend" }, moments.length ? "Move along the trace, or use the slider, to step both players through the approach." : ""))),
    el("ul", { class: "facts" }, facts.map((f) => el("li", {}, f))),
    el("div", { class: "goto" }, el("code", {}, k.demo_command), copy),
    el("p", { class: "muted", style: "margin-top:0.5rem" }, "Paste it into the CS2 console with this demo open to watch the approach."));
}

/**
 * The kill from above: both players' trails through the approach, where each of
 * them is looking (a pointer on the dot, and a cone of view in their side's
 * colour), and the line between them (solid when they could see each other,
 * dashed through a wall), on the map drawn from where players walk
 * (render_radars.py). show(i) moves everything to the i-th moment of the path.
 *
 * Stacked maps (Nuke, Vertigo) have a radar per floor. The one shown is where the
 * attacker stood at the shot; a toggle switches, and whatever is on the other
 * floor (a player, a stretch of trail) is drawn faded.
 */
let radarCount = 0;

function radarView(k, meta, mapName, shooter) {
  const NS = "http://www.w3.org/2000/svg";
  const path = k.path || [];
  const slider = el("input", { type: "range", min: 0, max: Math.max(0, path.length - 1), step: 1,
    "aria-label": "Moment in the approach" });
  const caption = el("figcaption", {});
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "The kill seen from above");
  const levels = meta?.levels;
  const floorOf = (z) => (levels && z != null ? (z >= levels.split_z ? "upper" : "lower") : null);
  const atShot = path.find((p) => p.ms === 0) || path[path.length - 1];
  let floor = floorOf(atShot?.az) || "upper";
  const buttons = levels ? ["upper", "lower"].map((f) => el("button", { type: "button", "data-floor": f }, f === "upper" ? "Upper" : "Lower")) : [];
  const toggle = levels ? el("div", { class: "floors", role: "group", "aria-label": "Floor" }, buttons) : null;
  const figure = el("figure", { class: "radar" }, svg, toggle, slider, caption);
  if (!path.length) {
    caption.textContent = "No positions were recorded for this kill.";
    return { figure, slider, show: () => {} };
  }

  // world units -> radar pixels (or plain world units when the map has no radar)
  const upp = meta ? meta.units_per_pixel : 1;
  const X = (x) => (meta ? (x - meta.x_min) / upp : x);
  const Y = (y) => (meta ? (meta.y_max - y) / upp : -y);
  const xs = path.flatMap((p) => [X(p.ax), p.vx == null ? null : X(p.vx)]).filter((v) => v != null);
  const ys = path.flatMap((p) => [Y(p.ay), p.vy == null ? null : Y(p.vy)]).filter((v) => v != null);
  const MIN_VIEW = 1400 / upp; // at least ~35 m across
  // room around both players for their cones of view
  const size = Math.max(MIN_VIEW, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 1.5;
  const cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
  svg.setAttribute("viewBox", `${cx - size / 2} ${cy - size / 2} ${size} ${size}`);
  const u = size / 340; // one screen pixel, roughly, at the usual panel width

  const add = (tag, attrs) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    svg.append(node);
    return node;
  };
  const image = meta ? add("image", { x: 0, y: 0, width: meta.size, height: meta.size }) : null;
  const colour = (side) => (side === "CT" ? "var(--ct)" : side === "T" ? "var(--t)" : "var(--graphite)");
  const away = (z) => levels != null && floorOf(z) != null && floorOf(z) !== floor;
  // a trail is one line per stretch on one floor, so the other floor's stretches can fade
  const stretches = [];
  const trail = (key, side, width, opacity) => {
    let run = null;
    for (const p of path) {
      if (p[`${key}x`] == null) continue;
      const point = `${X(p[`${key}x`])},${Y(p[`${key}y`])}`;
      const on = floorOf(p[`${key}z`]);
      if (!run || run.on !== on) {
        run = { on, opacity, points: run ? [run.points[run.points.length - 1]] : [] };
        stretches.push(run);
        run.node = add("polyline", { fill: "none", stroke: colour(side), "stroke-width": width,
          "stroke-linecap": "round", "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke" });
      }
      run.points.push(point);
      run.node.setAttribute("points", run.points.join(" "));
    }
  };
  trail("v", k.victim_side, 2, 0.55);
  trail("a", k.attacker_side, 2.5, 0.85);
  const paintFloor = () => {
    if (image) image.setAttribute("href", `/radars/${levels ? levels[floor] : mapName}.png`);
    for (const s of stretches) s.node.setAttribute("opacity", s.on && levels && s.on !== floor ? s.opacity * 0.3 : s.opacity);
    for (const b of buttons) b.setAttribute("aria-pressed", String(b.dataset.floor === floor));
  };
  // where each player looks: a cone of view that fades with distance, like a torch
  // beam, in their side's colour (the attacker's stronger), and a pointer on the dot
  const radarId = `radar${++radarCount}`;
  const defs = add("defs", {});
  const cone = (who, side, strength) => {
    const beam = document.createElementNS(NS, "radialGradient");
    beam.id = `${radarId}-${who}`;
    beam.setAttribute("gradientUnits", "userSpaceOnUse");
    for (const [offset, opacity] of [[0, strength], [1, 0]]) {
      const stop = document.createElementNS(NS, "stop");
      stop.setAttribute("offset", offset);
      stop.setAttribute("style", `stop-color:${colour(side)};stop-opacity:${opacity}`);
      beam.append(stop);
    }
    defs.append(beam);
    return { beam, shape: add("path", { fill: `url(#${beam.id})` }) };
  };
  const victimCone = cone("victim", k.victim_side, 0.26);
  const attackerCone = cone("attacker", k.attacker_side, 0.34);
  const sight = add("line", { stroke: "var(--graphite)", "stroke-width": 1.75, "vector-effect": "non-scaling-stroke" });
  // a player is a teardrop pointing where they look (one shape, one outline, so the
  // point reads at a glance), or a plain dot when their view is not known
  const pointer = (side, r) => {
    const group = add("g", {});
    const a = (40 * Math.PI) / 180, tip = 2.5 * r;
    const style = { stroke: "var(--paper)", "stroke-width": 1.5, "stroke-linejoin": "round",
      "vector-effect": "non-scaling-stroke" };
    const make = (tag, attrs) => {
      const node = document.createElementNS(NS, tag);
      for (const [key, value] of Object.entries({ ...style, ...attrs })) node.setAttribute(key, value);
      group.append(node);
      return node;
    };
    const nose = make("path", { d: `M${tip},0 L${r * Math.cos(a)},${r * Math.sin(a)} ` +
      `A${r},${r} 0 1 1 ${r * Math.cos(a)},${-r * Math.sin(a)} Z` });
    const dot = make("circle", { r });
    return { group, nose, dot, side };
  };
  const victim = pointer(k.victim_side, 5 * u);
  const attacker = pointer(k.attacker_side, 6 * u);
  // each dot carries its player's name: colours alone mislead once teams have
  // swapped sides, since the timeline colours players by the side they started on
  const short = (name) => (name && name.length > 16 ? `${name.slice(0, 15)}…` : name || "");
  const tag = (name, side, weight) => {
    const text = add("text", { "font-size": 10.5 * u, "font-weight": weight, fill: colour(side),
      stroke: "var(--paper)", "stroke-width": 3, "paint-order": "stroke", "stroke-linejoin": "round",
      "vector-effect": "non-scaling-stroke" });
    text.textContent = short(name);
    return text;
  };
  const victimTag = tag(k.victim || "victim", k.victim_side, 600);
  const attackerTag = tag(shooter || "attacker", k.attacker_side, 700);
  const nameAt = (text, x, y) => { text.setAttribute("x", x + 9 * u); text.setAttribute("y", y - 8 * u); };
  // CS yaw: 0 along +x, counter-clockwise; the radar's y axis points down
  const place = (marker, x, y, yaw) => {
    marker.group.setAttribute("transform", `translate(${x} ${y}) rotate(${yaw == null ? 0 : -yaw})`);
    marker.nose.setAttribute("visibility", yaw == null ? "hidden" : "visible");
    marker.dot.setAttribute("visibility", yaw == null ? "visible" : "hidden");
  };
  const aim = (c, x, y, yaw, reach) => {
    if (yaw == null) return c.shape.setAttribute("visibility", "hidden");
    const half = (45 * Math.PI) / 180, t0 = (yaw * Math.PI) / 180;
    const pt = (t) => `${x + reach * Math.cos(t)},${y - reach * Math.sin(t)}`;
    c.shape.setAttribute("visibility", "visible");
    c.shape.setAttribute("d", `M${x},${y} L${pt(t0 - half)} A${reach},${reach} 0 0,0 ${pt(t0 + half)} Z`);
    c.beam.setAttribute("cx", x); c.beam.setAttribute("cy", y); c.beam.setAttribute("r", reach);
  };
  // how far a view points from the other player, 0-180 degrees
  const offBy = (yaw, fx, fy, tx, ty) => {
    const bearing = (Math.atan2(ty - fy, tx - fx) * 180) / Math.PI;
    return Math.abs(((yaw - bearing + 540) % 360) - 180);
  };

  // a 10 m scale bar, bottom left
  const metre = 39.37 / upp;
  const bx = cx - size / 2 + 12 * u, by = cy + size / 2 - 12 * u;
  add("line", { x1: bx, x2: bx + 10 * metre, y1: by, y2: by, stroke: "var(--graphite)", "stroke-width": 2, "vector-effect": "non-scaling-stroke" });
  const label = add("text", { x: bx, y: by - 7 * u, "font-size": 10 * u, fill: "var(--graphite)" });
  label.textContent = "10 m";

  const who = `${k.victim || "the victim"}`;
  const show = (i) => {
    const p = path[Math.max(0, Math.min(path.length - 1, i))];
    const ax = X(p.ax), ay = Y(p.ay);
    const attackerAway = away(p.az), victimAway = p.vx != null && away(p.vz);
    // on the other floor: hollow and faded
    const paint = (marker, side, hollow, faded) => {
      for (const shape of [marker.nose, marker.dot]) {
        shape.setAttribute("fill", hollow ? "var(--paper)" : colour(side));
        shape.setAttribute("stroke", hollow ? colour(side) : "var(--paper)");
      }
      marker.group.setAttribute("opacity", faded ? 0.6 : 1);
    };
    place(attacker, ax, ay, p.yaw);
    nameAt(attackerTag, ax, ay);
    attackerTag.setAttribute("opacity", attackerAway ? 0.6 : 1);
    paint(attacker, k.attacker_side, attackerAway, attackerAway);
    aim(attackerCone, ax, ay, p.yaw, size * 0.3);
    attackerCone.shape.setAttribute("opacity", attackerAway ? 0.4 : 1);
    sight.setAttribute("opacity", attackerAway && victimAway ? 0.35 : 1);
    const hasVictim = p.vx != null;
    const alive = p.ms <= 0;
    victim.group.setAttribute("visibility", hasVictim ? "visible" : "hidden");
    victimTag.setAttribute("visibility", hasVictim ? "visible" : "hidden");
    sight.setAttribute("visibility", hasVictim ? "visible" : "hidden");
    let facing = "";
    if (hasVictim) {
      const vx = X(p.vx), vy = Y(p.vy);
      // a dead player looks nowhere: after the shot, no pointer and no cone
      place(victim, vx, vy, alive ? p.vyaw : null);
      nameAt(victimTag, vx, vy);
      victimTag.setAttribute("opacity", victimAway ? 0.6 : 1);
      paint(victim, k.victim_side, !alive || victimAway, victimAway);
      aim(victimCone, vx, vy, alive ? p.vyaw : null, size * 0.2);
      victimCone.shape.setAttribute("opacity", victimAway ? 0.4 : 1);
      sight.setAttribute("x1", ax); sight.setAttribute("y1", ay);
      sight.setAttribute("x2", vx); sight.setAttribute("y2", vy);
      sight.setAttribute("stroke-dasharray", p.visible ? "" : "5 4");
      if (alive && p.yaw != null) {
        const off = Math.round(offBy(p.yaw, p.ax, p.ay, p.vx, p.vy));
        facing += ` ${shooter || "The attacker"} looks ${off}° off them`;
        if (p.vyaw != null) {
          const back = offBy(p.vyaw, p.vx, p.vy, p.ax, p.ay);
          facing += `; ${who} ${back < 45 ? "faces them" : back > 100 ? "faces away" : "is side-on to them"}`;
        }
        facing += ".";
      }
    } else {
      aim(victimCone, 0, 0, null, 0);
    }
    slider.value = i;
    const when = p.ms === 0 ? "At the shot" : p.ms < 0 ? `${(-p.ms / 1000).toFixed(2).replace(/0$/, "")} s before the shot` : `${(p.ms / 1000).toFixed(2).replace(/0$/, "")} s after`;
    const other = floor === "upper" ? "lower" : "upper";
    const elsewhere = attackerAway && victimAway ? "Both are" : attackerAway ? "The attacker is" : victimAway ? `${who} is` : null;
    caption.textContent = `${when}: ${who} ${p.visible ? "in sight" : "behind cover"}.` + facing +
      (elsewhere ? ` ${elsewhere} on the ${other} floor.` : "") +
      (meta ? "" : ` No radar for ${mapName || "this map"}, so positions only.`);
  };
  for (const b of buttons) {
    b.addEventListener("click", () => { floor = b.dataset.floor; paintFloor(); show(Number(slider.value)); });
  }
  paintFloor();
  return { figure, slider, show };
}

/** Crosshair-to-head distance over the last 1.5 s, hidden stretches hatched. */
function trace(points, shots = []) {
  if (!points || !points.length) return el("p", { class: "muted" }, "No trace was recorded for this kill.");
  const NS = "http://www.w3.org/2000/svg";
  // drawn at the width it will be shown, so its labels stay readable on a phone
  const wide = view.clientWidth >= 900; // beside the radar, or under it on a phone
  const W = Math.round(wide ? Math.min(560, view.clientWidth * 0.6 - 380) : Math.max(300, view.clientWidth - 32));
  const H = 220, L = 44, R = 12, T = 12, B = 30;
  const x0 = -1500, x1 = 250;
  const top = Math.min(90, Math.max(10, Math.ceil(Math.max(...points.map((p) => p.angle)) / 10) * 10));
  const X = (ms) => L + ((ms - x0) / (x1 - x0)) * (W - L - R);
  const Y = (a) => T + (1 - Math.min(a, top) / top) * (H - T - B);
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "trace");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "How far the crosshair was from the enemy's head over the last 1.5 seconds");
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  };
  points.forEach((p, i) => {
    if (p.visible) return;
    const a = X(p.ms), b = i + 1 < points.length ? X(points[i + 1].ms) : X(x1);
    add("rect", { x: a, y: T, width: Math.max(0, b - a), height: H - T - B, fill: "url(#hatch)" });
  });
  for (const a of [0, top / 2, top]) {
    add("line", { class: "axis", x1: L, x2: W - R, y1: Y(a), y2: Y(a) });
    add("text", { x: L - 6, y: Y(a) + 4, "text-anchor": "end" }, `${a}°`);
  }
  for (const [ms, label] of [[-1500, "1.5 s before"], [-1000, "1 s"], [-500, "0.5 s"], [0, "shot"]]) {
    add("text", { x: X(ms), y: H - 10, "text-anchor": ms === -1500 ? "start" : "middle" }, label);
  }
  add("line", { class: "shot", x1: X(0), x2: X(0), y1: T, y2: H - B });
  // every shot the attacker fired, as a tick on the time axis
  for (const ms of shots.filter((s) => s >= x0 && s <= x1)) {
    add("line", { class: "fire", x1: X(ms), x2: X(ms), y1: H - B - 9, y2: H - B });
  }
  add("path", { class: "line", d: points.map((p, i) => `${i ? "L" : "M"}${X(p.ms).toFixed(1)},${Y(p.angle).toFixed(1)}`).join("") });
  points.forEach((p) => add("circle", { class: "dot", cx: X(p.ms), cy: Y(p.angle), r: 2.6 }));
  const cursor = add("line", { class: "cursor", y1: T, y2: H - B });
  svg.cursor = (ms) => { cursor.setAttribute("x1", X(ms)); cursor.setAttribute("x2", X(ms)); };
  svg.onScrub = (move, leave) => {
    const at = (e) => {
      const box = svg.getBoundingClientRect();
      return x0 + (((e.clientX - box.left) / box.width) * W - L) / (W - L - R) * (x1 - x0);
    };
    svg.addEventListener("pointermove", (e) => move(at(e)));
    svg.addEventListener("pointerleave", leave);
  };
  return svg;
}

function playerPanel(p, data) {
  const lines = [];
  lines.push(el("h2", {}, p.name || p.player_id));
  lines.push(el("p", { class: "sub muted" },
    `Started as ${p.side || "an unknown side"}. ${p.kills} ${p.kills === 1 ? "kill" : "kills"}. Steam ID ${p.player_id}.`));

  const flaggedByScore = p.flag_reasons.some((r) => r.startsWith("behaviour score"));
  if (p.score != null && !flaggedByScore) {
    lines.push(el("p", {}, p.enough_kills
      ? `Behaviour score ${p.score.toFixed(2)}, higher than ${pct(p.clean_percentile)} of clean players.`
      : `Behaviour score ${p.score.toFixed(2)}, from too few kills to mean much.`));
  }
  if (p.flagged) {
    lines.push(el("p", {}, el("b", { class: "flag" }, "Flagged: "), p.flag_reasons.map(sentence).join(". ") + "."));
  } else {
    lines.push(el("p", { class: "muted" },
      `Not flagged: the score is below the line (${pct(data.flag_percentile ?? 0.9)} of clean players) and no hard limit was broken.`));
  }

  if (p.rule_findings.length) {
    lines.push(el("h3", {}, "Hard limits broken"));
    lines.push(el("ul", {}, p.rule_findings.slice(0, 6).map((e) =>
      el("li", { class: "finding" }, el("b", {}, `${sentence(e.severity)}. `), sentence(e.note),
        e.tick != null ? el("span", { class: "muted" }, ` At tick ${e.tick}.`) : null))));
  }

  lines.push(el("h3", {}, "The judge"));
  if (p.verdict) {
    const v = p.verdict;
    lines.push(el("p", {},
      el("span", { class: `verdict ${v.verdict}` }, `${sentence(v.verdict)}, ${v.probability}%`),
      v.verdict === "cheating" && v.cheat_type !== "none" ? ` Suspected ${v.cheat_type}.` : ""));
    lines.push(el("ul", {}, v.reasons.map((r) => el("li", {}, sentence(r)))));
    if (v.caveats.length) {
      lines.push(el("h3", {}, "What would change its mind"));
      lines.push(el("ul", {}, v.caveats.map((c) => el("li", {}, sentence(c)))));
    }
  } else if (p.judge_error) {
    lines.push(el("p", { class: "warning" }, `The judge could not finish: ${p.judge_error}`));
  } else {
    lines.push(el("p", { class: "muted" }, p.flagged
      ? "The judge was not run for this review."
      : "The judge only reads flagged players."));
  }
  if (p.judge_evidence) {
    lines.push(el("details", {}, el("summary", {}, "What the judge read"), el("pre", {}, p.judge_evidence)));
  }
  return el("aside", { class: "player" }, lines);
}

/* ============================================================ navigation */

let keyboard = null;

function go(changes) {
  const [path, query] = location.hash.slice(1).split("?");
  const params = new URLSearchParams(query || "");
  for (const [k, v] of Object.entries(changes)) {
    if (v === undefined || v === null) params.delete(k);
    else params.set(k, v);
  }
  location.hash = `#${path}?${params}`;
}

function navigate(e, id, players, player, kill) {
  if (!player || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
  const kills = player.kill_log;
  const at = kill ? kills.findIndex((k) => k.tick === kill.tick) : -1;
  if (e.key === "ArrowRight" && at < kills.length - 1) go({ k: kills[at + 1].tick });
  else if (e.key === "ArrowLeft" && at > 0) go({ k: kills[at - 1].tick });
  else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    const i = players.indexOf(player) + (e.key === "ArrowDown" ? 1 : -1);
    const next = players[i];
    if (next) go({ p: next.player_id, k: (next.top_kills[0] || next.kill_log[0] || {}).tick });
  } else return;
  e.preventDefault();
}

document.addEventListener("keydown", (e) => keyboard && keyboard(e));

async function route() {
  clearTimeout(polling);
  keyboard = null;
  const [path, query] = location.hash.slice(1).split("?");
  const parts = (path || "/").split("/").filter(Boolean);
  const params = new URLSearchParams(query || "");
  const scrollY = window.scrollY;
  if (parts[0] === "job") await job(parts[1]);
  else if (parts[0] === "report") {
    const same = view.dataset.report === parts[1];
    await report(parts[1], params);
    if (same) window.scrollTo(0, scrollY); // choosing a kill should not jump the page
  } else {
    delete view.dataset.report;
    await home();
  }
}

window.addEventListener("hashchange", route);
api("/api/status").then((s) => { if (s.version) document.getElementById("version").textContent = `Overwatch review ${s.version}. `; }).catch(() => {});
route();
