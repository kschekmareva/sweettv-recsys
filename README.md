# sweet.tv TV programme recommender (Kaggle, MAP@5)

Course project for a master's recommender-systems exam. Kaggle competition: `sweettv-tv-program-recommender`.
For each of the 1,260 users in `submission.csv`, predict five programmes (`tv_show_id`) they will watch in ISO weeks 31-42 of 2020, using their view log of weeks 11-30 and the TV schedules. The metric is MAP@5.

**Final submission: `outputs/submissions/submission_STEP7_all.csv`** - a LightGBM (lambdarank) reranker over the top-50 candidates of six scorers (H2, EW, PPR, MF, AR, P-air), refit on all validation users.
Kaggle: public 0.42914, private 0.42244 (global-popularity baseline: 0.28234 / 0.27809). All submitted files and their scores are in `outputs/submissions/SUBMISSIONS.md`.

## Data (not in the repository)

`data/` is git-ignored. Put these five files there (competition files, downloadable with `python download.py` after configuring your own Kaggle credentials; `kagglehub` prints the folder it downloaded to - copy the files into `data/`):

`dataset11-30.csv` (view log), `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv` (schedules), `submission.csv` (template).

No credentials or tokens are stored in the repository.

## Setup and how to run

```
pip install -r requirements.txt        # Python 3.11, exact versions pinned (see outputs/environment.md)
python src/eda.py                      # step 1: data exploration -> outputs/eda.md
python src/run_validation.py           # step 2: join, interaction table, protocol, baselines -> outputs/validation_step2.md, protocol.md
python src/run_step3.py                # step 3: MF (implicit ALS), content-based, popularity hybrids
python src/run_step4.py                # step 4: truth variants V300/V900/VHALF, PPR, association rules, popularity variants
python src/run_step5.py                # step 5: airing-level join, VTOP5 truth, H2 scorer, pool option, blends
python src/run_step6.py                # step 6: decision rule (VTOP5 primary), VTOP5-tuned blend
python src/run_step7.py                # step 7: ceiling diagnostics, EW scorer, LightGBM reranker
python src/run_step8.py                # step 8: reranker improvements (2-fold swap), not shipped
python src/run_step8b.py               # step 8b: seed bagging + rank fusion, not shipped
python src/run_step10.py               # step 10 (8c): H2 decay grid, multi-scale features, learning curve, not shipped
python src/run_step11.py               # step 11: learning-curve rule -> submission_STEP7_all.csv
python src/make_final.py --rebuild --out outputs/submissions/repro_STEP7_all.csv   # regenerate the final file from the raw data in one command
python src/make_examples.py            # outputs/examples.md, outputs/handcheck.md
python src/make_docs.py                # requirements.txt, outputs/environment.md, outputs/report_tables.md/.csv, Kaggle columns
```

Steps must be run in this order: later steps read the JSON reports of earlier ones (chosen hyperparameters, weights). The first run builds the join cache in `outputs/cache/` (git-ignored, about 80 s). `make_final.py` is self-contained (frozen hyperparameters inside), takes `--rebuild` to rebuild the cache and `--out` for the output path (it never overwrites the committed final file unless asked); see `outputs/reproducibility.md` for the hash comparison.

## Validation protocol (5 lines)

1. A view matches a schedule row on the same channel when the time intervals overlap (no timezone shift); only rows with `tv_show_id != 0` are items; weeks 11-20 use the 11-20 schedule, weeks 21-30 the 21-30 schedule.
2. Temporal split: features/models from weeks 11-20, truth from weeks 21-30; the candidate pool is the set of programmes airing in the target weeks; repeats of already watched programmes are allowed; exactly five distinct ids per user, popularity fill.
3. Primary truth VTOP5 (the user's five most-completed programmes: airings watched to >= 80%, ties by the global count of such airings); V300, V900 and VHALF (lenient definitions) are robustness checks; the definition follows the public notebook cited below and was not tuned to any leaderboard score.
4. Users with truth are split 50/50 (seed 42) into TUNE (all choices of hyperparameters and weights) and REPORT (results); later steps use a 2-fold swap (fit on TUNE, score REPORT, and vice versa) with paired bootstrap 95% CIs (2,000 resamples, seed 42).
5. Decisions were pre-declared (shipping rules in `outputs/validation_step5.md`, "Step 6 - decision rule", and in each step's report); the final model is the step-7 reranker refit on all validation users, applied to features from weeks 21-30 and the 31-42 schedule without re-fitting on test.

## Repository layout

- `src/` - code (`data.py` join and interactions, `metrics.py` MAP@5, `baselines.py`, `mf.py`, `cbf.py`, `ppr.py`, `assoc.py`, `h2.py`, `ew.py`, `hybrid.py`, `rerank.py`, `rerank2.py`, `run_*.py` one script per step).
- `outputs/` - step reports (`validation_step*.md/.json`), `protocol.md`, `report_tables.md/.csv` (consolidated tables), `examples.md` (two example users), `handcheck.md`, `environment.md` (versions, hardware, runtimes), `submissions/` (all submission files and `SUBMISSIONS.md`).

## Web demo

A local, library-free page to browse the frozen model: pick a user (search by id, or the two example users A and B), switch between **Validation** (features from weeks 11-20, VTOP5 truth from weeks 21-30, REPORT users only) and **Final run** (weeks 31-42), see the Top-5 of every method with truth hits highlighted, the user's history by category, MAP@5 per method and the Kaggle table. Every number is labelled "from code (JSON)" or "computed in browser". The "Run checks" button runs `web/checks.js` (7 checks that recompute AP@5/MAP@5, validate `submission_STEP7_all.csv` and redo the hand-check overlaps in JavaScript, independent of Python's numbers) and prints the results with `console.table`.

```
python src/export_web.py               # rebuilds the validation/final lists from the frozen code -> web/data/web_data.json (about 3 minutes)
python -m http.server 8000             # from the repository root
# open http://localhost:8000/web/
node web/checks_node.js                # optional: the same checks headless, without a browser
```

`web/data/` is git-ignored because it is derived from competition data; the page code (`web/*.html|js|css`) and `src/export_web.py` are committed.

## Citation

The airing-level completion truth (VTOP5) and the H2 own-history scorer follow the idea of the public Kaggle notebook by kinahem, "gornyaki final solution": https://www.kaggle.com/code/kinahem/gornyaki-final-solution (concept only; no code was copied).

## Use of AI

The code, analysis and reports in this repository were written by Claude Code (Anthropic) under my prompts - first with Claude Opus 5.5, then with Claude Sonnet 5.5. I specified the tasks, the validation protocol and the pre-declared decision rules, and I made the choices about what to ship and what to submit to Kaggle.
