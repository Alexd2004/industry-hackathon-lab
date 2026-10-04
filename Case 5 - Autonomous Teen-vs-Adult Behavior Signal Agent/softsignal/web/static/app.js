// SoftSignal console. Reads the server's JSON (softsignal/web/server.py), draws everything with inline SVG.
// No library and no network beyond the local server, so the Wi-Fi-off demo works.
"use strict";

const N_ROUNDS = 8;
const REPLAY_MS = 1200;
const SVGNS = "http://www.w3.org/2000/svg";

const S = {
  view: "dashboard",
  runs: [],
  runId: null, // null: no run selected (the page opens empty)
  round: null, // null: follow the newest landed round
  cap: 15,
  stat: null,
  ladder: null,
  reviews: {},
  filter: "all",
  listN: 25,
  job: null,
  follow: null, // run id to select once it shows up (a run just started here)
  node: "a5",
  replay: null,
  poll: null,
  lists: {}, // run id -> its own list payload (fetched once the run has finished)
};

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const pct = (v, d = 0) => (isNum(v) ? `${(v * 100).toFixed(d)}%` : "–");
const pts = (v) => Math.round(v * 1000) / 10;

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 202) throw Object.assign(new Error(body.error || res.statusText), { body, status: res.status });
  return body;
}
const post = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function svg(w, h) { return el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }); }
function text(parent, x, y, s, attrs = {}) {
  const t = el("text", { x, y, "font-family": "var(--sans)", "font-size": 12, fill: "#3b4250", ...attrs }, parent);
  t.textContent = s;
  return t;
}

// ---------- run and round selection ----------
function run() { return S.runs.find((r) => r.id === S.runId) || null; }
function rounds() { return (run()?.rounds || []).slice().sort((a, b) => a.round - b.round); }
function shownRound() {
  const rs = rounds();
  if (!rs.length) return null;
  const last = rs.length - 1;
  return rs[S.round == null ? last : Math.min(S.round, last)];
}
function capOf(row) { return isNum(row?.cap) ? row.cap : (S.stat?.policy?.cap_false_teen ?? 0.15); }
function overCap(ft, cap) { return isNum(ft) && ft > cap + 1e-9; }
function counts() { return S.stat?.test || { n: 900, teens: 450, adults: 450 }; }
function minAdults() { return S.stat?.policy?.min_audit_adults ?? 120; }

const ACTION_TEXT = { hold: "Hold", "re-tune": "Re-tune", promote: "Promote to ACTIVE", starter: "Starter rule" };
const actionText = (a) => ACTION_TEXT[a] || a || "–";
function shortReason(reason, max = 64) {
  if (!reason) return "";
  const first = String(reason).split(/(?<=[.;])\s/)[0].replace(/[.;]$/, "");
  return first.length > max ? first.slice(0, max - 1).trimEnd() + "…" : first;
}

function gateOf(row) {
  if (!row) return { key: "none", short: "–" };
  if (row.round === 0) return { key: "starter", short: "Starter" };
  const a2 = row.agents?.a2;
  if (!a2 || !a2.status) return { key: "rule", short: "Rule" };
  if (row.applied_source === "A2") return { key: "passed", short: "Passed" };
  if (a2.status === "FALLBACK") return { key: "fallback", short: "Fallback" };
  return { key: "blocked", short: "Blocked" };
}

// ---------- top bar ----------
function renderTop() {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === S.view));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${S.view}`));
  const sel = $("run-select");
  const opts = `<option value="">No run selected</option>` + S.runs.map((r) => {
    const state = r.complete ? "" : (S.job?.running && S.job.run === r.run ? " · running" : " · partial");
    return `<option value="${esc(r.id)}">${esc(r.label)} · ${esc(r.badge)}${esc(state)}</option>`;
  }).join("");
  if (sel.innerHTML !== opts) sel.innerHTML = opts;
  sel.value = run() ? run().id : "";
  const row = shownRound();
  const total = N_ROUNDS - 1;
  $("round-chip").textContent = row ? `Round ${row.round} of ${total}${row.round === total ? " complete" : ""}`
    : S.job?.running ? "Starting run…" : "No run yet";
}

// ---------- empty states (no run selected yet) ----------
function waitText() {
  return S.job?.running ? "The run is starting: loading data and the text model…"
    : "Waiting for a run. Start one from Training → Run loop, or pick a replay in the run picker.";
}

// ---------- dashboard ----------
function renderDashboard() {
  const rs = rounds(), row = shownRound(), c = counts();
  if (!row) {
    $("d-rec").innerHTML = `–<small>%</small>`;
    $("d-rec-of").textContent = `Of the ${c.teens} teens in the held-out test. Appears with round 0.`;
    $("d-rec-delta").textContent = "";
    $("d-ft").innerHTML = `–<small>%</small>`;
    $("d-ft-of").textContent = `Of the ${c.adults} adults in the held-out test. Appears with round 0.`;
    $("d-ft-cap").textContent = "";
    $("d-chart-tag").textContent = "rounds.csv";
    $("d-chart").replaceChildren(roundChart([], -1, capOf(null), S.job?.running ? "Starting the run…" : "Waiting for a run"));
    $("d-chart-caption").textContent = `Held-out test (${c.n} accounts) after each round. A point is added as each round of the loop finishes. ${waitText()}`;
    return;
  }
  const r0 = rs[0], cap = capOf(row);
  $("d-rec").innerHTML = `${Math.round(row.rec * 100)}<small>%</small>`;
  $("d-rec-of").textContent = `${Math.round(row.rec * c.teens)} of ${c.teens} teens in the held-out test`;
  const d = Math.round((row.rec - r0.rec) * 100);
  $("d-rec-delta").textContent = row.round === 0 ? "Round 0: the organizers' starter rule"
    : d > 0 ? `+${d} points since round 0` : d < 0 ? `${d} points since round 0` : "No change since round 0";
  $("d-ft").innerHTML = `${(row.ft * 100).toFixed(1)}<small>%</small>`;
  $("d-ft-of").textContent = `${Math.round(row.ft * c.adults)} of ${c.adults} adults in the held-out test`;
  $("d-ft-cap").textContent = overCap(row.ft, cap) ? `Over the ${pct(cap)} cap` : `Within the ${pct(cap)} cap`;
  $("d-chart-tag").textContent = run().source === "live" ? "rounds.csv" : run().source === "rule" ? "rounds_rule.csv" : "rounds_recorded.csv";
  $("d-chart").replaceChildren(roundChart(rs, row.round, cap));
  $("d-chart-caption").textContent = `Held-out test (${c.n} accounts) after each round, at the loop's ${pct(cap)} cap. ` +
    `While the loop is in SHADOW the starter rule stays live, so the lines move only after a promote.`;
  renderCapPanel();
}

function gridRow(cap) {
  return (S.stat?.ranked?.grid || []).find((g) => Math.abs(g.cap - cap / 100) < 1e-9) || null;
}

function renderCapPanel() {
  $("cap-value").textContent = `${S.cap}%`;
  $("cap-slider").value = S.cap;
  document.querySelectorAll(".cap-pct").forEach((n) => (n.textContent = pct(capOf(shownRound()))));
  const g = gridRow(S.cap);
  $("cap-readout").textContent = !run() ? "What the model catches at this cap shows once a run is selected." : g
    ? `At a ${S.cap}% cap the SoftSignal stack (fit on all 2,100 training accounts) catches ${pct(g.rec_flagged, 1)} of held-out teens and flags ${pct(g.ft_flagged, 1)} of adults. ${g.n_verify} accounts go to verification now.`
    : "No cap grid yet: run Rebuild models on the Results tab.";
}

function roundChart(rs, current, cap, msg) {
  const W = 820, H = 560, L = 44, R = 120, T = 14, B = 42;
  const s = svg(W, H);
  const x = (i) => L + (i * (W - L - R)) / (N_ROUNDS - 1);
  const y = (v) => T + (1 - v) * (H - T - B);
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "#e3e6ec" }, s);
    text(s, L - 8, y(v) + 4, `${v * 100}%`, { "text-anchor": "end", "font-size": 14, fill: "#6b7280" });
  }
  el("rect", { x: L, y: y(cap), width: W - L - R, height: y(0) - y(cap), fill: "#e9ebf0" }, s);
  el("line", { x1: L, x2: W - R, y1: y(cap), y2: y(cap), stroke: "#12151c", "stroke-dasharray": "4 4", "stroke-width": 1 }, s);
  for (let i = 0; i < N_ROUNDS; i++) {
    const label = i === 0 ? "Round 0" : i === N_ROUNDS - 1 ? "Round 7" : String(i);
    text(s, x(i), H - B + 22, label, { "text-anchor": "middle", "font-size": 14.5, fill: i === current ? "#12151c" : "#6b7280", "font-weight": i === current ? 600 : 400 });
  }
  if (msg) text(s, (L + W - R) / 2, y(0.6), msg, { "text-anchor": "middle", "font-size": 18, "font-weight": 600, fill: "#6b7280" });
  const upto = rs.filter((r) => r.round <= current);
  const series = [["rec", "#2742d6", "Teens caught"], ["ft", "#b85c1e", "Adults flagged"]];
  for (const [k, color, name] of series) {
    const pts_ = upto.filter((r) => isNum(r[k]));
    if (!pts_.length) continue;
    el("polyline", { points: pts_.map((r) => `${x(r.round)},${y(r[k])}`).join(" "), fill: "none", stroke: color, "stroke-width": 2.5, "stroke-linejoin": "round" }, s);
    for (const r of pts_) el("circle", { cx: x(r.round), cy: y(r[k]), r: 4, fill: color }, s);
    const last = pts_[pts_.length - 1];
    text(s, x(last.round) + 10, y(last[k]) + 4, name, { fill: color, "font-size": 14.5, "font-weight": 500 });
  }
  return s;
}

// ---------- training ----------
function renderTraining() {
  const rs = rounds(), row = shownRound();
  $("t-title").textContent = row ? `Round ${row.round}` : "Round –";
  const last = rs.length ? rs[rs.length - 1].round : -1;
  $("t-pills").innerHTML = Array.from({ length: N_ROUNDS }, (_, i) =>
    `<button class="round-pill${row && row.round === i ? " active" : ""}" data-round="${i}" ${i > last ? "disabled" : ""}>${i}</button>`).join("");
  const busy = !!S.job?.running;
  $("run-btn").disabled = busy;
  $("pipeline-btn").disabled = busy;
  $("replay-btn").textContent = S.replay ? "Stop replay" : "Replay run";
  $("replay-btn").disabled = rs.length < 2;
  if (!row) {
    $("t-banner").innerHTML = `
      <div class="banner-box banner-a2"><div class="banner-label">A2 Loop Controller proposed</div><div class="banner-value">Waiting for round 1</div></div>
      <div class="banner-box banner-rule"><div class="banner-label">Plain rule would do</div><div class="banner-value">–</div></div>
      <div class="banner-gate"><span class="gate-chip">Code gate</span><span class="gate-text">${esc(waitText())}</span></div>`;
    $("t-weights").innerHTML = `<p class="placeholder">The live model's signal weights appear with round 0.</p>` +
      Array.from({ length: 6 }, () => `<div class="weight"><div class="weight-top"><span class="placeholder">–</span><span class="pct placeholder">–</span><span class="chg"></span></div><div class="bar"></div></div>`).join("");
    $("t-weights-note").textContent = "";
    renderLoop(null);
    renderAgent(null);
    $("t-table").querySelector("tbody").innerHTML = `<tr><td colspan="4" class="placeholder">Rounds appear here as they finish.</td></tr>`;
    return;
  }
  renderBanner(row);
  renderWeights(rs, row);
  renderLoop(row);
  renderAgent(row);
  renderTestTable(rs, row);
}

function statusChip(st) {
  if (!st) return "";
  return `<span class="status status-${esc(st)}">${esc(st)}</span>`;
}

function renderBanner(row) {
  const a2 = row.agents?.a2, rule = row.rule_decision, gate = gateOf(row), r = run();
  let proposed;
  if (row.round === 0) proposed = "Starter rule, nothing to decide";
  else if (!a2 || !a2.status) proposed = r.kind === "rule" ? "Not run (rule-only loop)" : "No A2 decision";
  else if (a2.status === "FALLBACK") proposed = `Fell back (${(a2.fallback_reason || "no output").replace(/_/g, " ")})`;
  else if (a2.output && typeof a2.output === "object" && a2.output.action) {
    const why = shortReason(a2.output.reason);
    proposed = `${actionText(a2.output.action)}${why ? ", " + why.charAt(0).toLowerCase() + why.slice(1) : ""}`;
  } else proposed = `Fell back (${a2.fallback_reason || "no output"})`;
  const ruleAction = rule?.action || row.action;
  let ruleText = actionText(ruleAction);
  if (ruleAction === "hold") ruleText += `, ${row.n_audit_adults} of ${minAdults()} audit adults`;
  const gateText = {
    starter: "Round 0 is the organizers' starter rule (w=0.45, cutoff 0.50).",
    rule: "No agents in this run: loop.py decided.",
    passed: `${Object.keys(row.diff || {}).length ? "Differs from the plain rule." : "Same as the plain rule."} A2's decision was applied.`,
    fallback: `A2 fell back (${a2?.fallback_reason || "invalid"}). The rule's decision was applied.`,
    blocked: "The code guards kept the rule's decision.",
  }[gate.key] || "";
  const gateLabel = gate.key === "starter" || gate.key === "rule" ? gate.short : `Code gate: ${gate.short.toLowerCase()}`;
  $("t-banner").innerHTML = `
    <div class="banner-box banner-a2"><div class="banner-label">A2 Loop Controller proposed ${statusChip(a2?.status)}</div>
      <div class="banner-value" title="${esc(a2?.output?.reason || "")}">${esc(proposed)}</div></div>
    <div class="banner-box banner-rule"><div class="banner-label">Plain rule would do</div>
      <div class="banner-value">${esc(ruleText)}</div></div>
    <div class="banner-gate"><span class="gate-chip${gate.key === "blocked" || gate.key === "fallback" ? " blocked" : ""}">${esc(gateLabel)}</span>
      <span class="gate-text">${esc(gateText)} Mode ${esc(row.mode)}.</span></div>`;
}

function modelWeights(row) {
  return row?.mode === "ACTIVE" ? S.stat?.weights?.stack || [] : S.stat?.weights?.starter || [];
}

function renderWeights(rs, row) {
  const now = modelWeights(row);
  const prevRow = rs.find((r) => r.round === row.round - 1);
  const same = !prevRow || prevRow.mode === row.mode;
  const prev = prevRow ? modelWeights(prevRow) : now;
  const max = Math.max(...now.map((w) => w.share), 0.01);
  $("t-weights").innerHTML = now.map((w) => {
    const before = prev.find((p) => p.feature === w.feature);
    let chg = "no change", cls = "";
    if (!same) {
      const d = before ? Math.round((w.share - before.share) * 100) : null;
      chg = d == null ? "new" : d === 0 ? "no change" : `${d > 0 ? "+" : ""}${d} pts`;
      cls = d == null || d > 0 ? "up" : d < 0 ? "down" : "";
    }
    return `<div class="weight"><div class="weight-top"><span>${esc(capitalize(w.label))}</span><span class="pct">${Math.round(w.share * 100)}%</span><span class="chg ${cls}">${esc(chg)}</span></div>
      <div class="bar"><i class="fill-${w.group}" style="width:${(w.share / max) * 96}%"></i></div></div>`;
  }).join("");
  $("t-weights-note").textContent = row.mode === "ACTIVE"
    ? "SoftSignal stack is live: each signal's share of the mean |contribution| on the held-out accounts (full-train fit)."
    : "Starter blend is live (SHADOW): each rule's share of the top score. The stack is refit beside it until it earns a promote.";
}
const capitalize = (s) => String(s).charAt(0).toUpperCase() + String(s).slice(1);

const NODES = [
  { key: "a1", tag: "A1", name: "Drift Watcher", deg: -10 },
  { key: "a3", tag: "A3", name: "Error Analyst", deg: 50 },
  { key: "a2", tag: "A2", name: "Loop Controller", deg: 108 },
  { key: "gate", tag: "GATE", name: "Code gate", deg: 155 },
  { key: "model", tag: "MODEL", name: "Retrain + test", deg: 205 },
  { key: "a4", tag: "A4", name: "Verify-band Triager", deg: 255 },
  { key: "a5", tag: "A5", name: "Honesty Auditor", deg: 300 },
];
const DOT = { LIVE: "#1f8a4c", REPLAY: "#5b6b9a", FALLBACK: "#c98a12", passed: "#1f8a4c", blocked: "#b85c1e", fallback: "#c98a12", starter: "#9aa1ad", rule: "#9aa1ad" };

function nodeState(key, row) {
  if (!row) return { status: null, sub: "waiting" };
  if (key === "gate") { const g = gateOf(row); return { status: g.key, sub: g.short }; }
  if (key === "model") {
    const sub = { hold: "held", "re-tune": "refit", promote: "promoted", starter: "starter" }[row.action] || row.action;
    return { status: row.action === "hold" || row.action === "starter" ? null : "LIVE", sub };
  }
  const b = row.agents?.[key];
  return { status: b?.status || null, sub: b?.status || "not run" };
}

function renderLoop(row) {
  const W = 560, H = 392, cx = 280, cy = 196, R = 172;
  const s = svg(W, H);
  const defs = el("defs", {}, s);
  const m = el("marker", { id: "arr", viewBox: "0 0 10 10", refX: 5, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
  el("path", { d: "M0,0 L10,5 L0,10 z", fill: "#2742d6" }, m);
  el("circle", { cx, cy, r: R, fill: "none", stroke: "#2742d6", "stroke-width": 2 }, s);
  const at = (deg, r = R) => [cx + r * Math.sin((deg * Math.PI) / 180), cy - r * Math.cos((deg * Math.PI) / 180)];
  NODES.forEach((n, i) => { // an arrowhead halfway to the next node, pointing clockwise
    const next = NODES[(i + 1) % NODES.length].deg + (i === NODES.length - 1 ? 360 : 0);
    const mid = (n.deg + next) / 2;
    const [x1, y1] = at(mid - 1), [x2, y2] = at(mid + 1);
    el("line", { x1, y1, x2, y2, stroke: "#2742d6", "stroke-width": 2, "marker-end": "url(#arr)" }, s);
  });
  text(s, cx, cy - 4, row ? `Round ${row.round}` : "Round –", { "text-anchor": "middle", "font-size": 32, "font-weight": 600, fill: row ? "#12151c" : "#9aa1ad" });
  text(s, cx, cy + 20, row ? `${pct(row.rec)} caught, ${pct(row.ft, 1)} flagged` : "Waiting for a run", { "text-anchor": "middle", "font-size": 12.5, fill: "#3b4250" });
  for (const n of NODES) {
    const st = nodeState(n.key, row);
    const [x, y] = at(n.deg);
    const w = 142, h = 46;
    const g = el("g", { class: `node${S.node === n.key ? " selected" : ""}${st.status ? "" : " idle"}`, "data-node": n.key, tabindex: 0 }, s);
    el("rect", { x: x - w / 2, y: y - h / 2, width: w, height: h, rx: 6 }, g);
    text(g, x - w / 2 + 10, y - 6, n.tag, { "font-family": "var(--mono)", "font-size": 11, fill: "#3b4250" });
    text(g, x - w / 2 + 10, y + 13, n.name, { "font-size": 13.5, "font-weight": 600, fill: "#12151c" });
    if (st.status && DOT[st.status]) el("circle", { cx: x + w / 2 - 11, cy: y - 10, r: 4.5, fill: DOT[st.status] }, g);
    const title = el("title", {}, g);
    title.textContent = `${n.tag} ${n.name}: ${st.sub}`;
  }
  $("t-loop").replaceChildren(s);
}

function agentSummary(key, row) {
  const b = row.agents?.[key], o = b?.output;
  const fb = b?.status === "FALLBACK" && b.fallback_reason ? ` (fallback: ${b.fallback_reason})` : "";
  switch (key) {
    case "a1":
      if (!b) return "Not run in this round.";
      return `Drift: ${o?.drift ?? "–"}. ${o?.reason ?? ""}${fb}`;
    case "a2":
      if (!b || !b.status) return row.round === 0 ? "No decision at round 0: the starter rule is live." : "Not run in this run (rule only).";
      if (o && o.action) return `${actionText(o.action)} at a ${pct(o.cap)} cap. ${o.reason ?? ""}${fb}`;
      return `No usable decision${fb}. The rule's decision was applied.`;
    case "a3":
      if (!b) return "Not run in this round.";
      if (!o || o.status !== "ok" || !(o.patterns || []).length) return `Not enough labelled errors from earlier rounds to analyse yet${fb}.`;
      return o.patterns.map((p) => `${p.description} (${p.n_accounts} accounts)`).join(" · ") + fb;
    case "a4":
      if (!b || !b.status) return "Not run: while the loop is in SHADOW the starter blend picks the verify band and has no explanations to triage.";
      return `${o?.batch_reason ?? ""}${fb}`;
    case "a5": {
      if (!b) return "Not run in this round.";
      if (!Array.isArray(o) || !o.length) return `No claims checked${fb}.`;
      const tally = {};
      o.forEach((c) => (tally[c.verdict] = (tally[c.verdict] || 0) + 1));
      const parts = Object.entries(tally).map(([v, n]) => `${n} ${v}`).join(", ");
      const bad = o.find((c) => c.verdict !== "supported");
      return `${o.length} claim${o.length > 1 ? "s" : ""} checked: ${parts}. ${bad ? `"${bad.claim}" → ${bad.note || bad.verdict}` : o[0].note || ""}${b.fallback_reason === "script_only" ? " (script check, no model call needed)" : fb}`;
    }
    case "gate": {
      const g = gateOf(row);
      return { starter: "Round 0: nothing to gate.", rule: "No A2 proposal: the rule's decision goes straight through.",
        passed: `A2's proposal passed the code guards (cap clamped to 8–30%, no refit before ${minAdults()} audit adults, no promote unless the pooled test passed) and was applied.`,
        fallback: "A2's output failed validation or timed out; the rule's decision was applied.",
        blocked: "A2's proposal broke a guard; the rule's decision was applied." }[g.key];
    }
    case "model": {
      const refit = isNum(row.refit_s) ? ` Refit took ${row.refit_s.toFixed(1)} s.` : "";
      return `${actionText(row.action)}. Mode ${row.mode}, verify cutoff ${isNum(row.t_verify) ? row.t_verify.toFixed(3) : "–"}. ${row.n_labels} labels revealed so far (${row.n_audit_adults} audit adults).${refit}`;
    }
  }
  return "";
}

function renderAgent(row) {
  const n = NODES.find((x) => x.key === S.node) || NODES[NODES.length - 1];
  if (!row) {
    $("t-agent").innerHTML = `<div class="agent-badge">${esc(n.tag.slice(0, 5))}</div>
      <div><div class="agent-title">${esc(n.name)}</div><div class="agent-text">Waiting for a run. This step's output shows here when its round lands. Click any step in the loop to follow it.</div></div>`;
    return;
  }
  const b = row.agents?.[n.key];
  const kind = n.key.startsWith("a") ? (b?.status === "LIVE" || b?.status === "REPLAY" ? "LLM" : b?.status || "") : "code";
  $("t-agent").innerHTML = `<div class="agent-badge">${esc(n.tag.slice(0, 5))}</div>
    <div><div class="agent-title">${esc(n.name)} ${kind ? `<span class="tag">${esc(kind)}</span>` : ""} ${statusChip(b?.status)}</div>
    <div class="agent-text">${esc(agentSummary(n.key, row))}</div></div>`;
}

function renderTestTable(rs, row) {
  $("t-table").querySelector("tbody").innerHTML = rs.map((r) => {
    const cap = capOf(r), g = gateOf(r);
    return `<tr class="click${r.round === row.round ? " sel" : ""}" data-round="${r.round}"><td>${r.round}</td>
      <td class="num">${pct(r.rec)}</td><td class="num${overCap(r.ft, cap) ? " over" : ""}">${pct(r.ft, 1)}</td><td>${esc(g.short)}</td></tr>`;
  }).join("");
}

// ---------- results ----------
function renderResults() {
  const rs = rounds(), row = shownRound();
  renderPeople(row);
  renderBeforeAfter(rs[0], row);
  renderHeat(!row);
  renderList();
  renderLadder();
  renderLatency();
}

function personPath(x, y, s) { // head and shoulders in an s-by-s box
  return `M${x + s * 0.5},${y + s * 0.06} a${s * 0.17},${s * 0.17} 0 1,0 0.01,0 z M${x + s * 0.16},${y + s * 0.96} q0,-${s * 0.5} ${s * 0.34},-${s * 0.5} q${s * 0.34},0 ${s * 0.34},${s * 0.5} z`;
}

function renderPeople(row) {
  const c = counts(), cap = capOf(row);
  const teens = Math.round((100 * c.teens) / c.n), adults = 100 - teens;
  const caught = row ? Math.round(row.rec * teens) : 0, flagged = row ? Math.round(row.ft * adults) : 0;
  const groups = row
    ? [[caught, "#2742d6", "teens caught"], [teens - caught, "#9fb0f0", "teens missed"],
      [flagged, "#b85c1e", "adults wrongly flagged"], [adults - flagged, "#c9ced8", "adults left alone"]]
    : [[100, "#e6e9ef", ""]];
  $("r-people-sub").textContent = row ? `${teens} teens and ${adults} adults, at the ${pct(cap)} cap after round ${row.round}.` : waitText();
  $("r-people-tag").textContent = !row ? "rounds.csv" : run().source === "rule" ? "rounds_rule.csv" : run().source === "live" ? "rounds.csv" : "rounds_recorded.csv";
  const cols = 20, size = 13, gap = 5;
  const s = svg(cols * (size + gap), 5 * (size + gap));
  let i = 0;
  for (const [n, color] of groups) {
    for (let k = 0; k < n; k++, i++) {
      const x = (i % cols) * (size + gap), y = Math.floor(i / cols) * (size + gap);
      el("path", { d: personPath(x, y, size), fill: color }, s);
    }
  }
  $("r-people").replaceChildren(s);
  const stats = row ? groups : [["–", "#e6e9ef", "teens caught"], ["–", "#e6e9ef", "teens missed"], ["–", "#e6e9ef", "adults wrongly flagged"], ["–", "#e6e9ef", "adults left alone"]];
  $("r-people-stats").innerHTML = stats.map(([n, color, label]) =>
    `<div class="pstat"><i style="background:${color}"></i><b>${n}</b><span>${esc(label)}</span></div>`).join("");
}

function renderHeat(waiting) {
  const h = S.stat?.heatmap;
  if (!h || !h.teen) { $("r-heat").innerHTML = `<p class="muted">No hourly sample in data/.</p>`; return; }
  const max = waiting ? Infinity : Math.max(...[...h.teen, ...h.adult].flat().filter(isNum), 0.01); // empty grid until a run
  const block = (grid, rgb, label, axis) => {
    const L = 34, cw = 12, ch = 10, W = L + 24 * cw, H = 7 * ch + (axis ? 18 : 2);
    const s = svg(W, H);
    const days = ["Mon", "", "Wed", "", "Fri", "", "Sun"];
    grid.forEach((rowv, d) => {
      if (days[d]) text(s, 0, d * ch + ch - 1, days[d], { "font-size": 9.5, fill: "#3b4250" });
      rowv.forEach((v, hr) => {
        const t = isNum(v) ? v / max : 0;
        el("rect", { x: L + hr * cw + 0.5, y: d * ch + 0.5, width: cw - 1, height: ch - 1, rx: 1.5, fill: `rgba(${rgb},${(0.06 + 0.94 * t).toFixed(3)})` }, s);
      });
    });
    if (axis) [[0, "12 am"], [6, "6 am"], [12, "12 pm"], [18, "6 pm"]].forEach(([hr, lab]) =>
      text(s, L + hr * cw, 7 * ch + 13, lab, { "font-size": 9.5, fill: "#6b7280" }));
    return `<div class="heat"><div class="heat-label" style="color:rgb(${rgb})">${esc(label)}</div>${s.outerHTML}</div>`;
  };
  $("r-heat").innerHTML = block(h.teen, "39,66,214", "Teens, 13 to 17", false) + block(h.adult, "184,92,30", "Adults, 23 and over", true) +
    `<p class="caption">${waiting ? "Fills in with the first run: when teens and adults are active, from the hourly sample."
      : `Mean sessions per hour, ${h.n_users.teen} teens and ${h.n_users.adult} adults in the hourly sample.`}</p>`;
}

function renderBeforeAfter(r0, row) {
  const cap = capOf(row);
  if (!row) {
    $("r-ba-sub").textContent = "Round 0 against the latest round, both at the loop's cap.";
    $("r-ba-tag").textContent = "rounds.csv";
    const empty = (label) => `<div class="ba-bar"><span>${label}</span><div class="bar"></div></div>`;
    const block = (title) => `<div class="ba-block"><div class="ba-title">${title}</div>
      <div class="ba-nums"><span class="placeholder">–%</span><span class="arrow">→</span><span class="placeholder">–%</span></div>${empty("Round 0")}${empty("Latest")}</div>`;
    $("r-ba").innerHTML = block("Teens caught") + block("Adults wrongly flagged") + `<p class="caption">${esc(waitText())}</p>`;
    return;
  }
  $("r-ba-sub").textContent = `Round 0 to round ${row.round}, both at the ${pct(cap)} cap.`;
  $("r-ba-tag").textContent = $("r-people-tag").textContent;
  const dRec = Math.round((row.rec - r0.rec) * 100), dFt = Math.round((r0.ft - row.ft) * 1000) / 10;
  const bar = (label, v, color, mark) => `<div class="ba-bar"><span>${label}</span><div class="bar"><i style="width:${Math.min(v, 1) * 100}%;background:${color}"></i>${mark != null ? `<span class="cap-mark" style="left:${mark * 100}%"></span>` : ""}</div></div>`;
  const status = (r) => (overCap(r.ft, cap) ? "over it" : "under");
  const scale = Math.max(2 * cap, r0.ft, row.ft) * 1.1; // false-teen bars: the cap marker lands near mid-track
  $("r-ba").innerHTML = `
    <div class="ba-block"><div class="ba-title">Teens caught</div>
      <div class="ba-nums"><span class="muted">${pct(r0.rec)}</span><span class="arrow">→</span><span class="blue">${pct(row.rec)}</span>
      <span class="change">${dRec === 0 ? "no change" : `${Math.abs(dRec)} points ${dRec > 0 ? "more" : "fewer"}`}</span></div>
      ${bar("Round 0", r0.rec, "#7d8696")}${bar(`Round ${row.round}`, row.rec, "#2742d6")}</div>
    <div class="ba-block"><div class="ba-title">Adults wrongly flagged</div>
      <div class="ba-nums"><span class="muted">${pct(r0.ft, 1)}</span><span class="arrow">→</span><span class="orange">${pct(row.ft, 1)}</span>
      <span class="change">${dFt === 0 ? "no change" : `${Math.abs(dFt)} points ${dFt > 0 ? "fewer" : "more"}`}</span></div>
      ${bar("Round 0", r0.ft / scale, "#7d8696", cap / scale)}${bar(`Round ${row.round}`, row.ft / scale, "#b85c1e", cap / scale)}</div>
    <p class="caption">| marks the ${pct(cap)} cap. Round 0 was ${status(r0)}, round ${row.round} is ${status(row)}.</p>`;
}

function runList() { return run() ? S.lists[run().id] : null; }

async function loadList() {
  const r = run();
  if (!r || !r.complete || S.lists[r.id] !== undefined || (S.job?.running && S.job.run === r.run)) return;
  S.lists[r.id] = null; // in flight
  try {
    S.lists[r.id] = await api(`/api/list?run=${encodeURIComponent(r.id)}`);
  } catch (e) { // asked once per run, never retried on every render
    S.lists[r.id] = { rows: null, error: e.status === 404 ? "the server is older than this page: restart python -m softsignal.web" : e.message };
  }
  render();
}

const ACTION_CHIP = { verify: "Request age verification", soft: "Teen-safe defaults", none: "No action" };
function why(r) {
  const chips = [r.c1, r.c2, r.c3].filter(Boolean).map((c) => String(c).replace(/\s*[+\-−]\d+(\.\d+)?\s*$/, ""));
  let s = chips.length ? capitalize(chips.join(", ")) + "." : "";
  if (r.band !== "none" && r.words) s += ` Teen-leaning words: ${r.words}.`;
  return s;
}

function renderList() {
  const r = run(), got = runList();
  if (!r) {
    $("r-filters").innerHTML = "";
    $("r-list-note").textContent = "";
    $("r-reviewed").textContent = `Reviewer · ${Object.keys(S.reviews).length} reviewed`;
    $("r-more").hidden = true;
    $("r-list").querySelector("tbody").innerHTML = `<tr><td colspan="6" class="placeholder">The likely-teen list appears when a run finishes, scored by the run's final model.</td></tr>`;
    return;
  }
  const rows = got?.rows || [];
  const n = { all: rows.length, verify: 0, soft: 0, none: 0 };
  rows.forEach((x) => n[x.band]++);
  const f = [["all", `All ${n.all.toLocaleString()}`], ["verify", `Request age verification ${n.verify}`], ["soft", `Teen-safe defaults ${n.soft}`], ["none", `No action ${n.none.toLocaleString()}`]];
  $("r-filters").innerHTML = rows.length ? f.map(([k, label]) => `<button class="filter${S.filter === k ? " active" : ""}" data-filter="${k}">${esc(label)}</button>`).join("") : "";
  const cap = pct(capOf(shownRound()));
  $("r-list-note").textContent = !rows.length ? "" : got.model === "stack"
    ? `Scored by this run's final model (the stack it promoted), banded at its own ${cap}-cap thresholds. ${got.file}`
    : `This run ended on the starter blend (no promote), so its list is the starter's: "why" is the rules that fired, with their weight in the score. ${got.file}`;
  $("r-reviewed").textContent = `Reviewer · ${Object.keys(S.reviews).length} reviewed`;
  const body = $("r-list").querySelector("tbody");
  $("r-more").hidden = true;
  if (!rows.length) {
    const msg = !r.complete ? "The list appears when the run finishes, scored by the run's final model."
      : got === null || got === undefined ? "Loading the run's list…"
      : got.error ? `Could not load the run's list: ${got.error}.`
      : "No list for this run: its final model was not saved. Start a rule-only run from Training → Run loop (a few seconds) to get one.";
    body.innerHTML = `<tr><td colspan="6" class="muted">${esc(msg)}</td></tr>`;
    return;
  }
  const shown = rows.filter((x) => S.filter === "all" || x.band === S.filter);
  body.innerHTML = shown.slice(0, S.listN).map((x) => {
    const v = S.reviews[x.blogger_id];
    return `<tr><td>${x.rank}</td><td>${esc(x.blogger_id)}</td>
      <td><div class="score">${x.score.toFixed(2)}<div class="bar"><i style="width:${Math.min(x.score, 1) * 100}%"></i></div></div></td>
      <td><span class="action action-${x.band}">${ACTION_CHIP[x.band]}</span></td>
      <td class="why">${esc(why(x))}</td>
      <td><div class="review" data-account="${esc(x.blogger_id)}"><button data-verdict="agree" class="${v === "agree" ? "on-agree" : ""}">Agree</button><button data-verdict="disagree" class="${v === "disagree" ? "on-disagree" : ""}">Disagree</button></div></td></tr>`;
  }).join("");
  $("r-more").hidden = shown.length <= S.listN;
}

function renderLadder() {
  const body = $("r-ladder").querySelector("tbody");
  if (!run()) { body.innerHTML = `<tr><td colspan="7" class="placeholder">The ladder appears with a run: its final round sits on top of the reference rows.</td></tr>`; return; }
  if (!S.ladder) return;
  const cap = S.stat?.policy?.cap_false_teen ?? 0.15;
  const rs = rounds(), last = rs[rs.length - 1];
  const mine = last ? [{ label: `This run, round ${last.round}${run().complete ? " (final)" : " (so far)"}`, rec: last.rec, ft: last.ft, prec: last.prec,
    f1: last.prec + last.rec ? (2 * last.prec * last.rec) / (last.prec + last.rec) : 0, auc: last.auc, source: run().label, mine: true }] : [];
  body.innerHTML = [...mine, ...S.ladder].map((r) => r.eval_set === "error"
    ? `<tr><td>${esc(r.label)}</td><td colspan="5" class="muted">could not compute</td><td>${esc(r.source)}</td></tr>`
    : `<tr class="${r.mine ? "this-run" : ""}"><td>${esc(r.label)}</td><td class="num">${pct(r.rec, 1)}</td><td class="num${overCap(r.ft, cap) ? " over" : ""}">${pct(r.ft, 1)}</td>
      <td class="num">${pct(r.prec, 1)}</td><td class="num">${isNum(r.f1) ? r.f1.toFixed(3) : "–"}</td><td class="num">${isNum(r.auc) ? r.auc.toFixed(3) : "–"}</td><td>${esc(r.source)}</td></tr>`).join("");
}

function renderLatency() {
  if (!run()) { $("r-latency").querySelector("tbody").innerHTML = `<tr><td colspan="7" class="placeholder">Measured agent times appear after a run.</td></tr>`; return; }
  const rows = S.stat?.latency || [];
  const ms = (v) => (isNum(v) ? (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`) : "–");
  $("r-latency").querySelector("tbody").innerHTML = rows.length ? rows.map((r) =>
    `<tr><td>${esc(r.row)}</td><td class="num">${r.n ?? "–"}</td><td class="num">${ms(r.p50_ms)}</td><td class="num">${ms(r.p95_ms)}</td>
     <td class="num">${isNum(r.tokens_in) ? `${r.tokens_in.toLocaleString()} / ${(r.tokens_out ?? 0).toLocaleString()}` : "–"}</td>
     <td class="num">${r.n_model_fallback ?? "–"}</td><td>${esc(r.note || "")}</td></tr>`).join("")
    : `<tr><td colspan="7" class="muted">No latency table yet.</td></tr>`;
}

// ---------- job toast ----------
const JOB_TITLE = { crew: "Crew run", rule: "Rule-only run", pipeline: "Rebuild models" };
function renderToast() {
  const j = S.job;
  if (!j || !j.kind || (!j.running && !S.toastSticky)) { $("toast").hidden = true; return; }
  $("toast").hidden = false;
  $("toast-title").textContent = `${JOB_TITLE[j.kind] || j.kind}${j.running ? " · running" : j.error ? " · failed" : " · done"}`;
  $("toast-step").textContent = j.error || j.step || "";
  $("toast-log").textContent = (j.log || []).join("\n");
  $("toast-log").scrollTop = 1e9;
}

// ---------- render all ----------
function render() {
  renderTop();
  if (S.view === "dashboard") renderDashboard();
  if (S.view === "training") renderTraining();
  if (S.view === "results") { renderResults(); loadList(); }
  renderCapPanel();
  renderToast();
}

// ---------- data ----------
async function loadRuns() {
  const got = await api("/api/runs");
  S.runs = got.runs;
  const wasRunning = S.job?.running;
  S.job = got.job;
  if (S.follow) {
    const r = S.runs.find((x) => x.source === "live" && x.run === S.follow);
    if (r) { S.runId = r.id; S.round = null; }
    if (!S.job.running) S.follow = null;
  }
  if (S.runId && !S.runs.find((r) => r.id === S.runId)) S.runId = null;
  if (wasRunning && !S.job.running && S.job.run) delete S.lists[`live:${S.job.run}`]; // its list was written at the end
  if (wasRunning && !S.job.running && S.job.kind === "pipeline") await loadStatic(true);
  schedulePoll();
}

async function loadStatic(withLadder) {
  S.stat = await api("/api/static");
  if (withLadder) { S.ladder = null; loadLadder(); }
}

async function loadLadder() {
  try { S.ladder = (await api("/api/ladder")).rows; } catch (e) { S.ladder = [{ label: "Ladder", eval_set: "error", source: e.message }]; }
  render();
}

function schedulePoll() {
  clearTimeout(S.poll);
  if (S.job?.running) S.poll = setTimeout(async () => { try { await loadRuns(); } catch { /* server restarting */ } render(); }, 1000);
}

async function startJob(kind, mode) {
  try {
    S.job = kind === "pipeline" ? await post("/api/pipeline") : await post("/api/run", { mode });
    S.toastSticky = true;
    if (kind !== "pipeline") S.follow = S.job.run;
    stopReplay();
  } catch (e) {
    S.job = e.body?.job || { kind, error: e.message, log: [] };
    S.toastSticky = true;
  }
  await loadRuns().catch(() => {});
  render();
}

// ---------- replay ----------
function selectRun(id) {
  stopReplay();
  S.runId = id;
  S.round = null;
  S.filter = "all";
  S.listN = 25;
  const r = run();
  if (r && r.source !== "live") toggleReplay(); // a committed run plays back from round 0
  else render();
}

function stopReplay() { clearInterval(S.replay); S.replay = null; }
function toggleReplay() {
  if (S.replay) { stopReplay(); render(); return; }
  const last = rounds().length - 1;
  S.round = 0;
  S.replay = setInterval(() => {
    if (S.round >= last) { stopReplay(); } else S.round += 1;
    render();
  }, REPLAY_MS);
  render();
}

// ---------- events ----------
function setView(v) {
  S.view = ["dashboard", "training", "results"].includes(v) ? v : "dashboard";
  if (location.hash !== `#${S.view}`) history.replaceState(null, "", `#${S.view}`);
  render();
}

function wire() {
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  window.addEventListener("hashchange", () => setView(location.hash.slice(1)));
  $("run-select").addEventListener("change", (e) => selectRun(e.target.value || null));
  $("cap-slider").addEventListener("input", (e) => { S.cap = Number(e.target.value); render(); });

  $("cap-default").addEventListener("click", () => { S.cap = Math.round((S.stat?.policy?.cap_false_teen ?? 0.15) * 100); render(); });
  $("t-pills").addEventListener("click", (e) => { const b = e.target.closest("[data-round]"); if (b && !b.disabled) { stopReplay(); S.round = Number(b.dataset.round); render(); } });
  $("t-table").addEventListener("click", (e) => { const tr = e.target.closest("[data-round]"); if (tr) { stopReplay(); S.round = Number(tr.dataset.round); render(); } });
  $("t-loop").addEventListener("click", (e) => { const g = e.target.closest("[data-node]"); if (g) { S.node = g.dataset.node; render(); } });
  $("replay-btn").addEventListener("click", toggleReplay);
  $("run-btn").addEventListener("click", (e) => { e.stopPropagation(); $("run-menu").hidden = !$("run-menu").hidden; });
  $("run-menu").addEventListener("click", (e) => { const b = e.target.closest("[data-mode]"); if (b) { $("run-menu").hidden = true; startJob("run", b.dataset.mode); } });
  document.addEventListener("click", () => ($("run-menu").hidden = true));
  $("pipeline-btn").addEventListener("click", () => startJob("pipeline"));
  $("toast-close").addEventListener("click", () => { S.toastSticky = false; $("toast").hidden = true; });
  $("r-filters").addEventListener("click", (e) => { const b = e.target.closest("[data-filter]"); if (b) { S.filter = b.dataset.filter; S.listN = 25; render(); } });
  $("r-more").addEventListener("click", () => { S.listN += 50; render(); });
  $("r-list").addEventListener("click", async (e) => {
    const b = e.target.closest("[data-verdict]");
    if (!b) return;
    const account = b.parentElement.dataset.account;
    const verdict = S.reviews[account] === b.dataset.verdict ? null : b.dataset.verdict;
    try { S.reviews = await post("/api/review", { account, verdict }); } catch { /* keep the old state */ }
    render();
  });
}

async function init() {
  wire();
  const asked = new URLSearchParams(location.search).get("run"); // ?run=rule | recorded | live | a full run id
  try {
    await Promise.all([loadStatic(false), loadRuns()]);
    S.cap = Math.round((S.stat.policy?.cap_false_teen ?? 0.15) * 100);
    const hit = asked && S.runs.find((r) => r.id === asked || r.source === asked || r.run === asked);
    if (hit) S.runId = hit.id; // a link can open a run directly; otherwise the page opens on no run
    S.reviews = await api("/api/reviews").catch(() => ({}));
  } catch (e) {
    document.querySelector("main").insertAdjacentHTML("afterbegin", `<div class="error-box">Could not load data from the server: ${esc(e.message)}</div>`);
  }
  setView(location.hash.slice(1) || "dashboard");
  loadLadder();
}

init();
