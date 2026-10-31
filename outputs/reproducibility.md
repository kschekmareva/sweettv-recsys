# Reproducibility of the final submission

`python src/make_final.py --rebuild --out outputs/submissions/repro_STEP7_all.csv` rebuilds the join cache from ./data, refits the LightGBM reranker (seed 42, deterministic, 4 threads, 142 trees, all 4,643 validation users with VTOP5 truth) and applies it to the final features.

- SHA-256 submission_STEP7_all.csv: `0ebca31a6acbf136df9747f6f78ebdd59d8c41a73a041f0a3c07ca1e664c7430`
- SHA-256 repro_STEP7_all.csv:      `0ebca31a6acbf136df9747f6f78ebdd59d8c41a73a041f0a3c07ca1e664c7430`
- byte-identical: **True**; identical lines: 1261 of 1261 (header + 1,260 users) = 100%
- runtime: 185.4 s including the cache rebuild (wall clock 188.3 s)

A first run exposed a bug in make_final.py (numpy cast the 64-bit user ids to float64 when selecting the users to fit on, so only 15 users were fitted); it was fixed (set membership + an assertion that 4,643 users are fitted) and the run repeated. The committed final file was produced by src/run_step11.py and was not affected.
