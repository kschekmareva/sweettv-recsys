# Reproducibility of submission_STEP15.csv

`python src/make_final.py --step15 --rebuild --out outputs/submissions/repro_STEP15.csv` (seed 42, deterministic LightGBM, 4 threads, 142 trees, fitted on all 4,643 validation users with VTOP5 truth, assertion in the script).

- SHA-256 of `submission_STEP15.csv`: `0da5664ced2867c396abe7f0a10ee0138588e00ebb40e471536406c177c3015e`
- SHA-256 of the regenerated file: `0da5664ced2867c396abe7f0a10ee0138588e00ebb40e471536406c177c3015e` (identical)
- Identical lines: 1,261 of 1,261 (100%); runtime 230.0 s including the join-cache rebuild.
- The regenerated file was deleted after the comparison.
