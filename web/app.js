"use strict";
(function () {
  const $ = id => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const FINAL = "step-7 reranker";
  const BADGE = { "P-air": "popularity" };
  let D, mode = "val", uid = null;

  const shortTitle = t => { const s = String(t).split(/[.,]/)[0].trim(); return s || String(t); };
  const hue = c => { let h = 0; for (const ch of c) h = (h * 31 + ch.charCodeAt(0)) % 360; return h; };
  const users = () => (mode === "val" ? Object.keys(D.validation.users) : D.final.template);

  function card(id, o) {
    o = o || {};
    const p = D.programmes[String(id)] || ["?", "?"];
    const c = el("div", "card" + (o.hit ? " hit" : "")), t = el("div", "t");
    if (o.hit) t.append(el("span", "chk", "✓ "));
    t.append(shortTitle(p[0]));
    c.title = p[0];
    c.append(t);
    const tags = el("div", "tags"), cat = el("span", "cat", p[1]);
    cat.style.background = "hsl(" + hue(p[1]) + ",45%,42%)";
    tags.append(cat);
    if (o.seen !== undefined) tags.append(el("span", "seen", o.seen ? "seen before" : "new"));
    if (o.extra) tags.append(el("span", "seen", o.extra));
    c.append(tags);
    if (o.badges && o.badges.length) { const b = el("div", "badges"); o.badges.forEach(n => b.append(el("span", "badge", BADGE[n] || n))); c.append(b); }
    return c;
  }

  // one method row: name, AP@5 (browser) with check mark, hits, cards
  function methodRow(m, ids, cls, ctx) {
    const row = el("div", "mrow " + cls), head = el("div", "rowhead");
    head.append(el("span", "nm", m === FINAL ? "Final model (step-7 reranker)" : m));
    const hitSet = ctx.truth;
    if (hitSet) {
      const ap = WebChecks.ap5(ids, ctx.truthIds), py = ctx.u.ap[m];
      const hits = new Set(ids.filter(i => hitSet.has(String(i)))).size;
      head.append(el("span", "ap", "AP@5 " + ap.toFixed(4)));
      const ok = el("span", "ok", Math.abs(ap - py) < 1e-9 ? "✓" : "✗ differs from Python");
      ok.title = Math.abs(ap - py) < 1e-9 ? "matches the Python value" : "does not match the Python value " + py.toFixed(4);
      head.append(ok, el("span", "muted", "hits: " + hits + " of 5"));
    }
    const cs = el("div", "cards");
    ids.forEach((id, k) => cs.append(card(id, {
      hit: hitSet && hitSet.has(String(id)), seen: ctx.seen.has(id),
      badges: m === FINAL ? ctx.from[k] : null })));
    row.append(head, cs);
    return row;
  }

  function renderSugg() {
    const q = $("q").value.trim(), ul = $("sugg");
    ul.replaceChildren();
    if (!q) { ul.hidden = true; return; }
    const hit = users().filter(u => u.includes(q)).slice(0, 8);
    hit.forEach(u => { const li = el("li", "", u); li.onmousedown = e => { e.preventDefault(); select(u); }; ul.append(li); });
    if (!hit.length) ul.append(el("li", "muted", "No user in this mode matches"));
    ul.hidden = false;
  }

  function profileText() {
    const h = D.history[uid] || { "11-20": {}, "21-30": {} };
    const wk = mode === "val" ? ["11-20"] : ["11-20", "21-30"], label = mode === "val" ? "weeks 11-20" : "weeks 11-30";
    const tot = {}; let n = 0;
    wk.forEach(w => Object.entries(h[w]).forEach(([c, k]) => { tot[c] = (tot[c] || 0) + k; n += k; }));
    if (!n) return "No completed airings in " + label + ".";
    const top = Object.entries(tot).sort((a, b) => b[1] - a[1])[0];
    return n + " completed airings in " + label + ", " + Math.round(100 * top[1] / n) + "% " + top[0];
  }

  function render() {
    $("m-val").classList.toggle("on", mode === "val"); $("m-fin").classList.toggle("on", mode === "fin");
    $("modehint").textContent = mode === "val" ? "Features from weeks 11-20, answers from weeks 21-30 (validation users not used for tuning)." : "Features from earlier weeks, recommendations for weeks 31-42.";
    $("utitle").textContent = "User " + uid;
    $("uprofile").textContent = profileText();
    renderHist();
    const tb = $("truthbox"), lg = $("legend"), fin = $("final"), oth = $("others");
    tb.replaceChildren(); lg.replaceChildren(); fin.replaceChildren(); oth.replaceChildren();
    let ctx, lists;
    if (mode === "val") {
      const u = D.validation.users[uid], truthIds = u.truth.map(x => String(x[0]));
      ctx = { u, truth: new Set(truthIds), truthIds, seen: new Set(u.seen), from: u.rr_from };
      lists = u.top;
      const box = el("div", "truthbox");
      box.append(el("h3", "", "What this user actually watched most in weeks 21-30 (the answer we try to guess)"));
      const cs = el("div", "cards");
      u.truth.forEach(x => cs.append(card(x[0], { seen: ctx.seen.has(x[0]), extra: x[1] + " completed" })));
      box.append(cs); tb.append(box);
      const l1 = el("div"), s1 = el("span", "sw g"); l1.append(s1, "Green card with a check mark: the user actually watched this among their top-5 programmes in weeks 21-30 (a hit).");
      const l2 = el("div"), s2 = el("span", "sw"); l2.append(s2, "White card: not among them.");
      lg.append(l1, l2, el("div", "", "Grey tag “seen before”: the user watched this programme in the feature weeks (11-20). Tag “new”: not watched there."));
    } else {
      const i = D.final.template.indexOf(uid), sv = new Set(D.final.seen[i]);
      ctx = { u: null, truth: null, seen: sv, from: D.final.rr_from[i] };
      lists = {}; D.final.methods.forEach(m => { lists[m] = D.final.lists[m][i]; });
      lg.append(el("div", "", "Weeks 31-42 are the future: no answers are available, so no hits are shown."),
                el("div", "", "Grey tag “seen before”: the user watched this programme in weeks 11-30. Tag “new”: not watched there."));
    }
    fin.append(methodRow(FINAL, lists[FINAL], "big final", ctx));
    D.meta.methods.filter(m => m !== FINAL).forEach(m => oth.append(methodRow(m, lists[m], "compact", ctx)));
  }

  function renderHist() {
    const box = $("hist"); box.replaceChildren();
    const h = D.history[uid];
    if (!h) { box.textContent = "No history for this user."; return; }
    const cats = Array.from(new Set([...Object.keys(h["11-20"]), ...Object.keys(h["21-30"])]));
    cats.sort((a, b) => (h["11-20"][b] || 0) + (h["21-30"][b] || 0) - (h["11-20"][a] || 0) - (h["21-30"][a] || 0));
    if (!cats.length) { box.append(el("p", "muted", "No completed airings in either window.")); return; }
    const mx = Math.max(1, ...cats.map(c => Math.max(h["11-20"][c] || 0, h["21-30"][c] || 0)));
    const grid = el("div", "bars");
    cats.forEach(c => [["11-20", "b1"], ["21-30", "b2"]].forEach(([w, cls], k) => {
      const n = h[w][c] || 0, bar = el("div", "bar " + cls);
      bar.style.width = (100 * n / mx) + "%";
      grid.append(el("div", "", k === 0 ? c : ""), bar, el("div", "muted", String(n)));
    }));
    box.append(grid);
  }

  function renderTable() {
    const mp = WebChecks.mapPerMethod(D), t = $("maptab"), n = Object.keys(D.validation.users).length;
    const head = el("tr");
    [["Method", ""], ["MAP@5 (" + n.toLocaleString("en-US") + " validation users)", "n"], ["matches Python", ""]].forEach(([x, c]) => head.append(el("th", c, x)));
    t.append(head);
    D.meta.methods.slice().sort((a, b) => mp[b] - mp[a]).forEach(m => {
      const r = el("tr", m === FINAL ? "fin" : ""), same = Math.abs(mp[m] - D.validation.map_python[m]) < 1e-9;
      const ok = el("td", "ok", same ? "✓" : "✗"); ok.title = same ? "matches the Python value" : "differs from Python " + D.validation.map_python[m].toFixed(4);
      r.append(el("td", "", m === FINAL ? "Final model (step-7 reranker)" : m), el("td", "n", mp[m].toFixed(4)), ok);
      t.append(r);
    });
  }

  function select(u) { uid = u; $("q").value = ""; $("sugg").hidden = true; render(); }
  function setMode(m) { mode = m; if (!users().includes(uid)) uid = users()[0]; render(); }

  async function run() {
    const csv = () => fetch("../outputs/submissions/submission_STEP7_all.csv").then(x => { if (!x.ok) throw new Error("HTTP " + x.status); return x.text(); });
    const r = await WebChecks.runChecks(D, csv, console);
    $("checkout").textContent = r.res.map(x => x.result + "  " + x.check + "  " + x.what + "\n        " + x.detail).join("\n") + "\n\n" + r.summary;
    window.__webChecks = r;
  }

  fetch("data/web_data.json").then(r => { if (!r.ok) throw new Error("web/data/web_data.json missing - run: python src/export_web.py"); return r.json(); }).then(d => {
    D = d; renderTable();
    $("m-val").onclick = () => setMode("val"); $("m-fin").onclick = () => setMode("fin");
    ["A", "B"].forEach(t => $("ex-" + t).onclick = () => { mode = "val"; select(D.examples[t]); });
    $("rnd").onclick = () => { const l = users(); select(l[Math.floor(Math.random() * l.length)]); };
    $("q").oninput = renderSugg;
    $("q").onblur = () => { $("sugg").hidden = true; };
    $("q").onkeydown = e => { if (e.key === "Enter") { const f = $("sugg").querySelector("li:not(.muted)"); if (f) select(f.textContent); } };
    $("run").onclick = () => run().catch(e => { $("checkout").textContent = "ERROR: " + e.message; });
    mode = "val"; uid = D.examples.A; render();
    if (/[?&]run\b/.test(location.search)) { $("ver").open = true; $("run").click(); }   // ?run runs the checks on load (headless run)
  }).catch(e => { $("utitle").textContent = String(e.message || e); });
})();
