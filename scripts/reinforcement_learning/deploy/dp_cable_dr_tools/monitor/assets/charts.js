// Minimal SVG charts: line chart with crosshair tooltip, and sparkline.
// Series: {name, points: [[x, y], ...], color: CSS var name like "--series-1"}.

const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function niceTicks(lo, hi, count = 4) {
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (lo === hi) { const d = Math.abs(lo) * 0.1 || 1; lo -= d; hi += d; }
  const step0 = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const ticks = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  if (ticks[ticks.length - 1] < hi) ticks.push(+(ticks[ticks.length - 1] + step).toPrecision(12));
  return ticks;
}

export function fmt(v, digits = 3) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(1) + "K";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(Math.max(0, digits - 1));
  if (a === 0) return "0";
  if (a < 1e-3) return v.toExponential(1);
  return v.toPrecision(digits);
}

let tooltipEl;
function tooltip() {
  if (!tooltipEl) {
    tooltipEl = document.createElement("div");
    tooltipEl.className = "tooltip";
    document.body.appendChild(tooltipEl);
  }
  return tooltipEl;
}

function showTooltip(evt, xLabel, rows) {
  const t = tooltip();
  t.replaceChildren();
  const x = document.createElement("div");
  x.className = "x";
  x.textContent = xLabel;
  t.appendChild(x);
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "row";
    const key = document.createElement("span");
    key.className = "key";
    key.style.background = `var(${r.color})`;
    const strong = document.createElement("strong");
    strong.textContent = r.value;
    row.append(key, strong);
    if (r.name) {
      const nm = document.createElement("span");
      nm.className = "nm";
      nm.textContent = r.name;
      row.appendChild(nm);
    }
    t.appendChild(row);
  }
  t.style.display = "block";
  const pad = 14;
  let left = evt.clientX + pad, top = evt.clientY + pad;
  const w = t.offsetWidth, h = t.offsetHeight;
  if (left + w > window.innerWidth - 8) left = evt.clientX - w - pad;
  if (top + h > window.innerHeight - 8) top = evt.clientY - h - pad;
  t.style.left = left + "px";
  t.style.top = top + "px";
}

function hideTooltip() { if (tooltipEl) tooltipEl.style.display = "none"; }

function nearestIndex(points, x) {
  let lo = 0, hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] < x) lo = mid; else hi = mid;
  }
  return Math.abs(points[lo][0] - x) <= Math.abs(points[hi][0] - x) ? lo : hi;
}

/**
 * Line chart. opts: {height, yDomain: [lo, hi] | null, xLabel: "iteration", yFormat, xMax, markers: [{x, label}]}
 */
export function lineChart(container, series, opts = {}) {
  container.replaceChildren();
  const width = Math.max(200, container.clientWidth || 320);
  const height = opts.height || 180;
  const m = { top: 8, right: 12, bottom: 22, left: 44 };
  const iw = width - m.left - m.right, ih = height - m.top - m.bottom;
  const all = series.flatMap((s) => s.points);
  const svg = el("svg", { width, height, role: "img" }, container);
  if (!all.length) {
    el("text", { x: width / 2, y: height / 2, "text-anchor": "middle" }, svg).textContent = "no data yet";
    return;
  }
  let [ylo, yhi] = opts.yDomain || [Math.min(...all.map((p) => p[1])), Math.max(...all.map((p) => p[1]))];
  const yt = niceTicks(ylo, yhi, 4);
  ylo = yt[0]; yhi = yt[yt.length - 1];
  const xlo = opts.xMin ?? Math.min(...all.map((p) => p[0]));
  const xhi = Math.max(opts.xMax ?? -Infinity, ...all.map((p) => p[0]));
  const xs = (x) => m.left + (xhi === xlo ? iw / 2 : ((x - xlo) / (xhi - xlo)) * iw);
  const ys = (y) => m.top + ih - ((y - ylo) / (yhi - ylo || 1)) * ih;
  const yFormat = opts.yFormat || ((v) => fmt(v));

  for (const t of yt) {
    el("line", { x1: m.left, x2: m.left + iw, y1: ys(t), y2: ys(t), stroke: cssVar("--grid"), "stroke-width": 1 }, svg);
    el("text", { x: m.left - 6, y: ys(t) + 4, "text-anchor": "end" }, svg).textContent = yFormat(t);
  }
  el("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih, stroke: cssVar("--axis"), "stroke-width": 1 }, svg);
  for (const t of niceTicks(xlo, xhi, Math.max(2, Math.floor(iw / 80)))) {
    if (t < xlo || t > xhi) continue;
    el("text", { x: xs(t), y: height - 6, "text-anchor": "middle" }, svg).textContent = fmt(t);
  }
  for (const mk of opts.markers || []) {
    if (mk.x < xlo || mk.x > xhi) continue;
    el("line", { x1: xs(mk.x), x2: xs(mk.x), y1: m.top, y2: m.top + ih, stroke: cssVar("--axis"), "stroke-width": 1 }, svg);
  }
  for (const s of series) {
    if (!s.points.length) continue;
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${xs(p[0]).toFixed(1)},${ys(p[1]).toFixed(1)}`).join("");
    el("path", { d, fill: "none", stroke: `var(${s.color})`, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    if (s.dots) {
      for (const p of s.points) {
        el("circle", { cx: xs(p[0]), cy: ys(p[1]), r: 4, fill: `var(${s.color})`, stroke: cssVar("--surface-1"), "stroke-width": 2 }, svg);
      }
    } else {
      const p = s.points[s.points.length - 1];
      el("circle", { cx: xs(p[0]), cy: ys(p[1]), r: 4, fill: `var(${s.color})`, stroke: cssVar("--surface-1"), "stroke-width": 2 }, svg);
    }
  }

  const cross = el("line", { y1: m.top, y2: m.top + ih, stroke: cssVar("--text-muted"), "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = series.map((s) => el("circle", { r: 4, fill: `var(${s.color})`, stroke: cssVar("--surface-1"), "stroke-width": 2, visibility: "hidden" }, svg));
  const hit = el("rect", { x: m.left, y: 0, width: iw, height, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (evt) => {
    const r = svg.getBoundingClientRect();
    const x = xlo + ((evt.clientX - r.left - m.left) / iw) * (xhi - xlo);
    let snapX = null;
    const rows = [];
    series.forEach((s, i) => {
      if (!s.points.length) { dots[i].setAttribute("visibility", "hidden"); return; }
      const p = s.points[nearestIndex(s.points, x)];
      if (snapX === null || Math.abs(p[0] - x) < Math.abs(snapX - x)) snapX = p[0];
      dots[i].setAttribute("cx", xs(p[0]));
      dots[i].setAttribute("cy", ys(p[1]));
      dots[i].setAttribute("visibility", "visible");
      rows.push({ name: series.length > 1 ? s.name : null, value: yFormat(p[1]), color: s.color, y: p[1] });
    });
    if (snapX === null) return;
    cross.setAttribute("x1", xs(snapX));
    cross.setAttribute("x2", xs(snapX));
    cross.setAttribute("visibility", "visible");
    rows.sort((a, b) => b.y - a.y);
    showTooltip(evt, `${opts.xLabel || "iteration"} ${fmt(snapX)}`, rows);
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
    hideTooltip();
  });
}

export function sparkline(points, opts = {}) {
  const w = opts.width || 80, h = opts.height || 22;
  const svg = el("svg", { width: w, height: h, "aria-hidden": "true" });
  if (!points || points.length < 2) return svg;
  let lo = opts.yDomain ? opts.yDomain[0] : Math.min(...points.map((p) => p[1]));
  let hi = opts.yDomain ? opts.yDomain[1] : Math.max(...points.map((p) => p[1]));
  if (hi === lo) { hi += 1; lo -= 1; }
  const x0 = points[0][0], x1 = points[points.length - 1][0];
  const xs = (x) => 2 + ((x - x0) / (x1 - x0 || 1)) * (w - 6);
  const ys = (y) => h - 3 - ((y - lo) / (hi - lo)) * (h - 6);
  const d = points.map((p, i) => `${i ? "L" : "M"}${xs(p[0]).toFixed(1)},${ys(p[1]).toFixed(1)}`).join("");
  el("path", { d, fill: "none", stroke: `var(${opts.color || "--series-1"})`, "stroke-width": 1.5, "stroke-linejoin": "round" }, svg);
  const p = points[points.length - 1];
  el("circle", { cx: xs(p[0]), cy: ys(p[1]), r: 2.5, fill: `var(${opts.color || "--series-1"})` }, svg);
  return svg;
}

const STATUS = {
  RUNNING: ["good", "▶", "Running"],
  COMPLETED: ["good", "✓", "Completed"],
  PENDING: ["warning", "…", "Pending"],
  WAITING: ["warning", "…", "Waiting"],
  SCHEDULING: ["warning", "…", "Scheduling"],
  INITIALIZING: ["warning", "…", "Initializing"],
  NOT_SUBMITTED: ["neutral", "–", "Not submitted"],
  UNKNOWN: ["neutral", "?", "Unknown"],
};

export function statusBadge(run) {
  let [cls, icon, label] = STATUS[run.status] || (
    /^(FAILED|ERROR|TIMEOUT)/.test(run.status || "") ? ["critical", "✕", titleCase(run.status)]
    : /^CANCEL/.test(run.status || "") ? ["serious", "■", "Cancelled"] : ["neutral", "?", run.status || "Unknown"]);
  if (run.gave_up) [cls, icon, label] = [ "critical", "✕", run.oom ? "Failed: OOM" : "Failed (gave up)"];
  else if (run.hold && cls !== "good") [cls, icon, label] = ["neutral", "‖", label + " (held)"];
  const span = document.createElement("span");
  span.className = `status ${cls}`;
  const i = document.createElement("span");
  i.className = "icon";
  i.textContent = icon;
  const t = document.createElement("span");
  t.textContent = label;
  span.append(i, t);
  if (run.priority === "HIGH") {
    const h = document.createElement("span");
    h.className = "tag";
    h.textContent = "HIGH";
    h.title = "HIGH priority: not preemptible; queues within the pool's HIGH/NORMAL quota";
    span.appendChild(h);
  }
  if ((run.attempts || 1) > 1) {
    const a = document.createElement("span");
    a.className = "muted";
    a.textContent = `↻${run.attempts - 1}`;
    a.title = `${run.attempts - 1} relaunch(es)`;
    span.appendChild(a);
  }
  return span;
}

function titleCase(s) { return s.charAt(0) + s.slice(1).toLowerCase().replace(/_/g, " "); }

export function ago(iso) {
  if (!iso) return "–";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return `${Math.round(s)}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 172800) return `${(s / 3600).toFixed(1)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export function pct(v) { return v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`; }
