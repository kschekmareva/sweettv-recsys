"use strict";
(function () {
  const $ = id => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const FINAL = "step-7 reranker";
  const BADGE = { "P-air": "popularity" };
  const DESC = {
    "P": "most watched by all users", "P-air": "popular and airs often in the target weeks", "H2": "what this user completed most, recent weeks weigh more",
    "CBF": "similar category, genre, director and actors", "MF": "what users with similar taste watch (matrix factorization)",
    "PPR": "random walk on the user-programme graph from this user", "AR": "programmes often watched together with this user's programmes",
    "step-6 blend": "weighted mix of P-air, MF, PPR, H2"};
  let D, mode = "val", uid = null;
  const infoCache = {};

  // 'ТСН. Неделя'. -> ТСН. Неделя ; "'Name'. Description." -> Name ; prefixes (М/с, Х/ф, Т/с) are kept; cut at 40 characters
  function shortTitle(t) {
    let s = String(t).trim();
    const m = s.match(/^['"«“](.+?)['"»”]\s*(?:[.,]|$)/);
    if (m) { s = m[1]; if ((s.match(/'/g) || []).length % 2) s += "'"; }
    else s = s.replace(/\.$/, "");
    return s.length > 40 ? s.slice(0, 39).replace(/[\s,.]+$/, "") + "…" : s;
  }
  const hue = c => { let h = 0; for (const ch of c) h = (h * 31 + ch.charCodeAt(0)) % 360; return h; };
  const users = () => (mode === "val" ? Object.keys(D.validation.users) : D.final.template);

  function card(id, o) {
    o = o || {};
    const p = D.programmes[String(id)] || ["?", "?"];
    const c = el("div", "card" + (o.hit ? " hit" : "")), t = el("div", "t");
    if (o.hit) t.append(el("span", "chk", "✓ "));
    t.append(shortTitle(p[0]));
    c.title = p[0];
    if (o.rank) c.append(el("span", "rk", "#" + o.rank));
    c.append(t);
    const tags = el("div", "tags"), cat = el("span", "cat", p[1]);
    cat.style.background = "hsl(" + hue(p[1]) + ",45%,42%)";
    tags.append(cat);
    if (o.seen !== undefined) tags.append(el("span", "seen", o.seen ? "seen before" : "new"));
    if (o.extra) tags.append(el("span", "seen", o.extra));
    c.append(tags);
    if (o.badges && o.badges.length) {
      const b = el("div", "badges");
      o.badges.forEach(([n, r]) => b.append(el("span", "badge", (BADGE[n] || n) + " #" + r)));
      c.append(b);
    }
    return c;
  }

  // one method row: name (+ description), AP@5 computed in the browser with a check mark, hits, five cards
  function methodRow(m, ids, cls, ctx) {
    const compact = cls.includes("compact");
    const row = el("div", "mrow " + cls), head = el("div", "rowhead");
    const nm = el("span", "nm", m === FINAL ? "Final model (step-7 reranker)" : m);
    if (compact && DESC[m]) nm.append(el("span", "desc", DESC[m]));
    head.append(nm);
    if (ctx.truth) {
      const ap = WebChecks.ap5(ids, ctx.truthIds), py = ctx.u.ap[m], same = Math.abs(ap - py) < 1e-9;
      const hits = new Set(ids.filter(i => ctx.truth.has(String(i)))).size;
      const stat = el("span", compact ? "stat" : "");
      const ok = el("span", "ok", same ? "✓" : "✗ differs from Python");
      ok.title = same ? "matches the Python value" : "does not match the Python value " + py.toFixed(4);
      stat.append(el("span", "ap", "AP@5 " + ap.toFixed(4) + " "), ok, el("span", "muted", " hits: " + hits + " of 5"));
      if (compact) head.append(stat); else head.append(...stat.childNodes);
      if (m === FINAL) ctx.fin = { hits, ap };
    }
    const cs = el("div", "cards");
    ids.forEach((id, k) => cs.append(card(id, {
      rank: k + 1, hit: ctx.truth && ctx.truth.has(String(id)), seen: ctx.seen.has(id), badges: m === FINAL ? ctx.from[k] : null })));
    row.append(head, cs);
    return row;
  }

  // dropdown of all users of the current mode: "<id> - N completed airings, mostly <category>", N descending
  function userInfo(u) {
    const h = D.history[u] || { "11-20": {}, "21-30": {} }, tot = {}; let n = 0;
    (mode === "val" ? ["11-20"] : ["11-20", "21-30"]).forEach(w => Object.entries(h[w]).forEach(([c, k]) => { tot[c] = (tot[c] || 0) + k; n += k; }));
    const top = Object.entries(tot).sort((a, b) => b[1] - a[1])[0];
    return { n, cat: top ? top[0] : null };
  }
  function fillSelect() {
    if (!infoCache[mode]) { const o = {}; users().forEach(u => { o[u] = userInfo(u); }); infoCache[mode] = o; }
    const info = infoCache[mode], q = $("q").value.trim(), sel = $("usel");
    const list = users().filter(u => !q || u.includes(q)).sort((a, b) => info[b].n - info[a].n);
    sel.replaceChildren();
    list.forEach(u => {
      const o = el("option", "", u + " - " + (info[u].n ? info[u].n + " completed airings, mostly " + info[u].cat : "no completed airings"));
      o.value = u; sel.append(o);
    });
    if (!list.length) { const o = el("option", "", "No user matches"); o.disabled = true; sel.append(o); }
    $("ucount").textContent = list.length + " of " + users().length + " users";
    if (list.includes(uid)) sel.value = uid;
    return list;
  }

  function profileText() {
    const i = infoCache[mode][uid] || userInfo(uid), label = mode === "val" ? "weeks 11-20" : "weeks 11-30";
    if (!i.n) return "No completed airings in " + label + ".";
    const h = D.history[uid], tot = {};
    (mode === "val" ? ["11-20"] : ["11-20", "21-30"]).forEach(w => Object.entries(h[w]).forEach(([c, k]) => { tot[c] = (tot[c] || 0) + k; }));
    return i.n + " completed airings in " + label + ", " + Math.round(100 * tot[i.cat] / i.n) + "% " + i.cat;
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
      u.truth.forEach((x, k) => cs.append(card(x[0], { rank: k + 1, seen: ctx.seen.has(x[0]), extra: x[1] + " completed" })));
      box.append(cs); tb.append(box);
      const l1 = el("div"), l2 = el("div");
      l1.append(el("span", "sw g"), "Green card with a check mark: the user actually watched this among their top-5 programmes in weeks 21-30 (a hit).");
      l2.append(el("span", "sw"), "White card: not among them.");
      lg.append(l1, l2, el("div", "", "Grey tag “seen before”: the user watched this programme in the feature weeks (11-20). Tag “new”: not watched there."));
    } else {
      const i = D.final.template.indexOf(uid);
      ctx = { u: null, truth: null, seen: new Set(D.final.seen[i]), from: D.final.rr_from[i] };
      lists = {}; D.final.methods.forEach(m => { lists[m] = D.final.lists[m][i]; });
      lg.append(el("div", "", "Weeks 31-42 are the future: no answers are available, so no hits are shown."),
                el("div", "", "Grey tag “seen before”: the user watched this programme in weeks 11-30. Tag “new”: not watched there."));
    }
    const frow = methodRow(FINAL, lists[FINAL], "big final", ctx);
    if (mode === "val") {
      fin.append(el("p", "summary", "The final model found " + ctx.fin.hits + " of the user's " + ctx.truthIds.length + " most-watched programmes (AP@5 = " + ctx.fin.ap.toFixed(2) + ")."));
    }
    fin.append(frow, el("p", "note", "Badges show which simple methods proposed this programme and at which rank in their own top-50; the final model re-ordered these candidates."));
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

  function select(u) { uid = u; $("q").value = ""; fillSelect(); render(); }
  function setMode(m) {
    mode = m; $("q").value = "";
    if (!users().includes(uid)) uid = users()[0];
    fillSelect(); render();
  }

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
    $("q").oninput = () => { const l = fillSelect(); if (l.length && !l.includes(uid)) { uid = l[0]; $("usel").value = uid; render(); } };
    $("usel").onchange = () => { uid = $("usel").value; render(); };
    $("run").onclick = () => run().catch(e => { $("checkout").textContent = "ERROR: " + e.message; });
    mode = "val"; uid = D.examples.A; fillSelect(); render();
    // headless-run hooks: ?run runs the checks, ?mode=fin starts in Final run, ?open opens the compare section
    const qs = location.search;
    if (/[?&]mode=fin\b/.test(qs)) { setMode("fin"); }
    if (/[?&]open\b/.test(qs)) $("cmp").open = true;
    if (/[?&]run\b/.test(qs)) { $("ver").open = true; $("run").click(); }
  }).catch(e => { $("utitle").textContent = String(e.message || e); });
})();
