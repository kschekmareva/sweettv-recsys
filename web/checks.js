/* Independent checks. They use only ids, truth lists, raw timestamps and the submission CSV; Python's numbers are used
   only as the values to compare against. Works in the browser (window.WebChecks) and in Node (module.exports). */
(function (root) {
  "use strict";

  // AP@5, standard definition: sum P@i * rel(i) / min(5, |truth|); a repeated id counts once; empty truth -> 0
  function ap5(pred, truth) {
    const t = new Set(truth.map(String)), seen = new Set();
    let hits = 0, s = 0;
    for (let i = 0; i < Math.min(5, pred.length); i++) {
      const p = String(pred[i]);
      if (t.has(p) && !seen.has(p)) { hits++; s += hits / (i + 1); }
      seen.add(p);
    }
    return t.size ? s / Math.min(5, t.size) : 0;
  }
  const fix4 = x => (Math.round(x * 1e4) / 1e4).toFixed(4);
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;

  function mapPerMethod(D) {
    const out = {};
    for (const m of D.meta.methods) {
      out[m] = mean(Object.values(D.validation.users).map(u => ap5(u.top[m], u.truth.map(x => x[0]))));
    }
    return out;
  }

  function parseCsv(text) {
    const lines = text.split(/\r?\n/).filter(l => l.length);
    return { header: lines[0], rows: lines.slice(1).map(l => { const i = l.indexOf(","); return [l.slice(0, i), l.slice(i + 1)]; }) };
  }
  function utc(s) { // "YYYY-MM-DD HH:MM:SS" or "DD.MM.YYYY HH:MM:SS" -> epoch seconds
    let m = s.match(/^(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)$/);
    if (m) return Date.UTC(+m[1], m[2] - 1, +m[3], +m[4], +m[5], +m[6]) / 1000;
    m = s.match(/^(\d\d)\.(\d\d)\.(\d{4}) (\d\d):(\d\d):(\d\d)$/);
    return Date.UTC(+m[3], m[2] - 1, +m[1], +m[4], +m[5], +m[6]) / 1000;
  }

  async function runChecks(D, loadCsv, log) {
    const res = [];
    const add = (id, what, ok, detail) => res.push({ check: id, what, result: ok ? "PASS" : "FAIL", detail });
    const U = D.validation.users, M = D.meta.methods;

    // C1 JS AP@5 == Python per-user AP@5, every REPORT user and method
    let n = 0, bad = [], maxd = 0;
    for (const [uid, u] of Object.entries(U)) for (const m of M) {
      n++;
      const d = Math.abs(ap5(u.top[m], u.truth.map(x => x[0])) - u.ap[m]);
      maxd = Math.max(maxd, d);
      if (!(d < 1e-9)) bad.push(uid + "/" + m + " diff " + d);
    }
    add("C1", "JS AP@5 = Python AP@5 (|diff| < 1e-9)", bad.length === 0,
        n + " user-method pairs, max |diff| = " + maxd.toExponential(2) + (bad.length ? "; mismatches: " + bad.slice(0, 5).join("; ") : "; 0 mismatches"));

    // C2 MAP@5 over REPORT users
    const mp = mapPerMethod(D);
    add("C2", "JS MAP@5: P = 0.2699, step-7 reranker = 0.4047", fix4(mp["P"]) === "0.2699" && fix4(mp["step-7 reranker"]) === "0.4047",
        "P " + fix4(mp["P"]) + ", step-7 reranker " + fix4(mp["step-7 reranker"]) + " over " + Object.keys(U).length + " users");

    // C3 / C4 the submission file
    const csv = parseCsv(await loadCsv());
    const pool = new Set(D.pool_final.map(String));
    const tmpl = D.final.template;
    const probs = [];
    if (csv.header !== "user_id,tv_show_id") probs.push("header '" + csv.header + "'");
    if (csv.rows.length !== 1260) probs.push("rows " + csv.rows.length);
    let orderBad = 0, distinctBad = 0, poolBad = 0;
    csv.rows.forEach((r, i) => {
      if (r[0] !== tmpl[i]) orderBad++;
      const ids = r[1].trim().split(/\s+/);
      if (ids.length !== 5 || new Set(ids).size !== 5) distinctBad++;
      if (!ids.every(x => pool.has(x))) poolBad++;
    });
    if (tmpl.length !== csv.rows.length) probs.push("template length " + tmpl.length);
    if (orderBad) probs.push(orderBad + " rows out of template order");
    if (distinctBad) probs.push(distinctBad + " rows without 5 distinct ids");
    if (poolBad) probs.push(poolBad + " rows with ids outside the final pool");
    add("C3", "submission_STEP7_all.csv: header, 1,260 rows, template order, 5 distinct ids, ids in final pool (" + D.pool_final.length + ")", probs.length === 0,
        probs.length ? probs.join("; ") : csv.rows.length + " rows, header ok, order ok, distinct ok, pool ok");

    let diff = 0;
    csv.rows.forEach((r, i) => { const a = r[1].trim().split(/\s+/).join(" "), b = (D.final.final_step7_all[i] || []).join(" "); if (a !== b) diff++; });
    add("C4", "every CSV row equals the final STEP7_all list in the JSON", diff === 0 && D.final.final_step7_all.length === csv.rows.length,
        diff + " differing rows of " + csv.rows.length);

    // C5 user A, step by step
    const A = D.examples.A, ua = U[A];
    const tset = new Set(ua.truth.map(x => String(x[0])));
    let hits = 0, sum = 0; const steps = [];
    ua.top["step-7 reranker"].forEach((id, i) => {
      const hit = tset.has(String(id));
      if (hit) { hits++; sum += hits / (i + 1); }
      steps.push({ k: i + 1, id, hit, hits_so_far: hits, "P@k at hit": hit ? hits + "/" + (i + 1) + " = " + (hits / (i + 1)).toFixed(4) : "-" });
    });
    const den = Math.min(5, tset.size), apA = sum / den;
    add("C5", "user A (" + A + "): AP@5 = " + fix4(apA) + " (expect 0.3333)", fix4(apA) === "0.3333",
        "hits at ranks " + steps.filter(s => s.hit).map(s => s.k).join(", ") + "; sum of P@k at hits = " + sum.toFixed(4) + "; denominator min(5, |truth| = " + tset.size + ") = " + den + "; AP@5 = " + apA.toFixed(4));
    if (log) log.table(steps);

    // C6 overlaps from raw timestamps
    const H = D.handcheck, vs = utc(H.view.start_time), ve = utc(H.view.stop_time);
    const expect = [[364, "0.3033"], [1429, "0.9527"]];
    let ok6 = H.schedule.length === 2; const det6 = [];
    H.schedule.forEach((r, i) => {
      const s = utc(r.start_time), e = s + r.duration;
      const ov = Math.max(0, Math.min(e, ve) - Math.max(s, vs)), pct = ov / r.duration;
      det6.push(ov + " s / " + fix4(pct));
      if (!expect[i] || ov !== expect[i][0] || fix4(pct) !== expect[i][1] || ov !== r.code_overlap_sec) ok6 = false;
    });
    add("C6", "hand-check overlaps from raw timestamps: 364 s / 0.3033 and 1,429 s / 0.9527", ok6,
        det6.join(" ; ") + " (seconds from code: " + H.schedule.map(r => r.code_overlap_sec).join(", ") + ")");

    // C7 the four unit cases of src/metrics.py
    const cases = [
      ["perfect list", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0],
      ["hits at k=2 and k=4, |truth|=2", [9, 1, 8, 2, 7], [1, 2], (1 / 2 + 2 / 4) / 2],
      ["single hit at k=5, |truth|=6", [1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 10], (1 / 5) / 5],
      ["empty truth", [1, 2, 3, 4, 5], [], 0.0]];
    const cr = cases.map(c => ({ case: c[0], got: ap5(c[1], c[2]), expected: c[3], ok: Math.abs(ap5(c[1], c[2]) - c[3]) < 1e-12 }));
    add("C7", "four MAP@5 unit cases of src/metrics.py", cr.every(c => c.ok), cr.map(c => c.case + " " + c.got.toFixed(4)).join("; "));
    if (log) log.table(cr);

    const pass = res.filter(r => r.result === "PASS").length, fail = res.length - pass;
    if (log) log.table(res);
    const summary = "WEB CHECKS - pass " + pass + ", fail " + fail;
    if (log) log.log(summary);
    return { res, pass, fail, summary };
  }

  const api = { runChecks, ap5, mapPerMethod, fix4, mean };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.WebChecks = api;
})(typeof window !== "undefined" ? window : globalThis);
