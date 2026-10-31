"use strict";
(function () {
  const $ = id => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const tagEl = kind => el("span", "tag " + (kind === "js" ? "js" : "code"), kind === "js" ? "computed in browser" : "from code (JSON)");
  let D, mode = "val", uid = null;

  function card(id, hit, extra) {
    const p = D.programmes[String(id)] || ["?", "?"];
    const c = el("div", "card" + (hit ? " hit" : ""));
    c.append(el("div", "id", String(id) + (hit ? "  HIT" : "") + (extra || "")), el("div", "t", p[0]), el("div", "c", p[1]));
    c.title = p[0];
    return c;
  }
  const users = () => (mode === "val" ? Object.keys(D.validation.users) : D.final.template);

  function renderList() {
    const q = $("q").value.trim(), all = users();
    const found = q ? all.filter(u => u.includes(q)) : all;
    $("count").textContent = found.length + " of " + all.length + " users" + (found.length > 60 ? " (first 60 shown)" : "");
    const ul = $("ulist"); ul.replaceChildren();
    found.slice(0, 60).forEach(u => { const li = el("li", u === uid ? "sel" : "", u); li.onclick = () => select(u); ul.append(li); });
  }

  function renderUser() {
    const g = $("grid"), tb = $("truthbox"); g.replaceChildren(); tb.replaceChildren();
    if (!uid) return;
    if (mode === "val") {
      const u = D.validation.users[uid], truth = u.truth.map(x => String(x[0])), ts = new Set(truth);
      $("uhead").textContent = "User " + uid + " - validation (features weeks 11-20, truth weeks 21-30)";
      const t = el("div", "truth"), h = el("div", "mname", "VTOP5 truth, with completed airings in weeks 21-30 ");
      h.append(tagEl("code"));
      const cs = el("div", "cards"); u.truth.forEach(x => cs.append(card(x[0], false, "  | " + x[1] + " completed")));
      t.append(h, cs); tb.append(t);
      D.meta.methods.forEach(m => {
        const row = el("div", "mrow"), name = el("div", "mname", m), s = el("small", "", "AP@5 " + WebChecks.ap5(u.top[m], truth).toFixed(4) + " ");
        s.append(tagEl("js"), el("br"), "Python " + u.ap[m].toFixed(4) + " ", tagEl("code"));
        name.append(s);
        const cs2 = el("div", "cards"); u.top[m].forEach(id => cs2.append(card(id, ts.has(String(id)))));
        row.append(name, cs2); g.append(row);
      });
    } else {
      $("uhead").textContent = "User " + uid + " - final run (weeks 31-42, no truth exists)";
      const i = D.final.template.indexOf(uid);
      const note = el("p", "muted", "Top-5 lists from code (JSON). Reranker row = submission_STEP7_all.csv (features from weeks 21-30); other rows use the history of weeks 11-30.");
      note.append(" ", tagEl("code"));
      tb.append(note);
      D.final.methods.forEach(m => {
        const row = el("div", "mrow"), name = el("div", "mname", m === "step-7 reranker" ? m + " (FINAL)" : m);
        const cs = el("div", "cards"); D.final.lists[m][i].forEach(id => cs.append(card(id, false)));
        row.append(name, cs); g.append(row);
      });
    }
    renderHist();
  }

  function renderHist() {
    const box = $("hist"); box.replaceChildren();
    const h = D.history[uid];
    if (!h) { box.textContent = "No history for this user."; return; }
    const cats = Array.from(new Set([...Object.keys(h["11-20"]), ...Object.keys(h["21-30"])]));
    cats.sort((a, b) => (h["11-20"][b] || 0) + (h["21-30"][b] || 0) - (h["11-20"][a] || 0) - (h["21-30"][a] || 0));
    const mx = Math.max(1, ...cats.map(c => Math.max(h["11-20"][c] || 0, h["21-30"][c] || 0)));
    const leg = el("div", "leg muted");
    const i1 = el("i", "b1"), i2 = el("i", "b2");
    leg.append(i1, "weeks 11-20", i2, "weeks 21-30");
    box.append(leg);
    if (!cats.length) { box.append(el("p", "muted", "No completed airings in either window.")); return; }
    const grid = el("div", "bars");
    cats.forEach(c => {
      [["11-20", "b1"], ["21-30", "b2"]].forEach(([w, cls], k) => {
        const n = h[w][c] || 0, bar = el("div", "bar " + cls);
        bar.style.width = (100 * n / mx) + "%";
        grid.append(el("div", "", k === 0 ? c : ""), bar, el("div", "muted", String(n)));
      });
    });
    box.append(grid);
  }

  function renderTables() {
    const mp = WebChecks.mapPerMethod(D), t = $("maptab");
    $("mapn").textContent = "(" + Object.keys(D.validation.users).length + " users with VTOP5 truth)";
    const head = el("tr"); ["method", "MAP@5 computed in browser", "MAP@5 from code (JSON)", "|diff|"].forEach(x => head.append(el("th", "", x)));
    t.append(head);
    D.meta.methods.forEach(m => {
      const r = el("tr"), py = D.validation.map_python[m];
      r.append(el("td", "", m), el("td", "n", mp[m].toFixed(4)), el("td", "n", py.toFixed(4)), el("td", "n", Math.abs(mp[m] - py).toExponential(1)));
      t.append(r);
    });
    const k = $("kaggle"), kh = el("tr"); ["file", "method", "public", "private"].forEach(x => kh.append(el("th", "", x))); k.append(kh);
    D.kaggle.forEach(x => { const r = el("tr"); r.append(el("td", "", x.file), el("td", "", x.method), el("td", "n", x.public), el("td", "n", x.private)); k.append(r); });
  }

  function select(u) { uid = u; renderList(); renderUser(); }
  function setMode(m) {
    mode = m; $("m-val").classList.toggle("on", m === "val"); $("m-fin").classList.toggle("on", m === "fin");
    if (!users().includes(uid)) uid = users()[0];
    renderList(); renderUser();
  }

  async function run() {
    const csv = () => fetch("../outputs/submissions/submission_STEP7_all.csv").then(x => { if (!x.ok) throw new Error("HTTP " + x.status); return x.text(); });
    const r = await WebChecks.runChecks(D, csv, console);
    $("checkout").textContent = r.res.map(x => x.result + "  " + x.check + "  " + x.what + "\n        " + x.detail).join("\n") + "\n\n" + r.summary;
    window.__webChecks = r;
  }

  fetch("data/web_data.json").then(r => { if (!r.ok) throw new Error("web/data/web_data.json missing - run: python src/export_web.py"); return r.json(); }).then(d => {
    D = d; renderTables();
    $("m-val").onclick = () => setMode("val"); $("m-fin").onclick = () => setMode("fin");
    ["A", "B"].forEach(t => {
      $("ex-" + t).textContent = "Example user " + t + " (" + D.examples[t] + ")";
      $("ex-" + t).onclick = () => { if (mode !== "val") setMode("val"); select(D.examples[t]); };
    });
    $("q").oninput = renderList;
    $("run").onclick = () => run().catch(e => { $("checkout").textContent = "ERROR: " + e.message; });
    select(D.examples.A);
    if (/[?&]run\b/.test(location.search)) $("run").click();   // http://localhost:8000/web/?run runs the checks on load (used for the headless run)
  }).catch(e => { $("uhead").textContent = String(e.message || e); });
})();
