# Environment and runtimes

- OS: Windows-10-10.0.19045-SP0 (Windows 10 Home Single Language, build 10.0.19045)
- Python: 3.11.9 (CPython)
- CPU: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel, 4 logical cores (LightGBM and PPR use 4 threads; BLAS is limited to 1 thread)
- RAM: 7.65 GB
- Packages (exact versions, also in requirements.txt): implicit==0.7.3, kagglehub==1.0.3, lightgbm==4.6.0, numpy==2.3.3, pandas==2.3.3, psutil==7.1.0, scikit-learn==1.7.2, scipy==1.16.2

Runtimes copied from the step reports (seconds, wall clock on this machine):

| step | seconds |
|---|---|
| 1 EDA (src/eda.py) | 72.4 |
| 2 interactions, protocol, baselines (src/run_validation.py; includes building the join cache) | 94.5 |
| 3 MF, CBF, hybrids (src/run_step3.py) | 169.3 |
| 4 truth variants, PPR, AR, hybrids (src/run_step4.py) | 626.0 |
| 5 VTOP5, H2, pool option, blends (src/run_step5.py) | 293.3 |
| 6 decision rule, VTOP5-tuned blend (src/run_step6.py) | 88.4 |
| 7 ceiling, EW, reranker (src/run_step7.py) | 221.7 |
| 8 reranker improvements, 2-fold swap (src/run_step8.py) | 743.4 |
| 8b seed bagging, rank fusion (src/run_step8b.py) | 180.4 |
| 10 (8c) H2 decay grid, multi-scale features, learning curve (src/run_step10.py) | 137.4 |
| 11 final refit on all validation users (src/run_step11.py) | 160.7 |
| 15 per-week reranker, short-window analogue and final file (src/run_step15.py) | 247.9 |
| 16 reproduction of submission_STEP15.csv, with cache rebuild (src/make_final.py --step15 --rebuild) | 230.0 |
| 9 reproduction of the final file, with cache rebuild (src/make_final.py --rebuild) | 185.4 |

The join cache (outputs/cache/interactions_v3.pkl) is rebuilt from ./data in about 80 s; later steps reuse it.
