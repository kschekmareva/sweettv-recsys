"""MAP@5, standard Kaggle definition.

AP@k = sum_{i=1..k} P@i * rel(i) / min(k, |truth|); empty truth -> 0.
A repeated id in a list is counted as a hit only once.
"""
import numpy as np


def rel_vector(pred, truth, k=5):
    seen, rel = set(), []
    for p in list(pred)[:k]:
        rel.append(int(p in truth and p not in seen))
        seen.add(p)
    return rel


def ap_at_k(pred, truth, k=5):
    if not truth:
        return 0.0
    hits, score = 0, 0.0
    for i, r in enumerate(rel_vector(pred, truth, k), 1):
        if r:
            hits += 1
            score += hits / i
    return score / min(k, len(truth))


def map_at_k(preds, truths, users, k=5):
    """Mean AP@k over `users`; users missing from `truths` have empty truth (AP = 0)."""
    return float(np.mean([ap_at_k(preds[u], truths.get(u, set()), k) for u in users]))


def ap5_matrix(top_idx, truth, k=5):
    """Vectorised AP@k. top_idx: (n, >=k) distinct column indices; truth: (n, n_items) bool matrix.
    Same definition as ap_at_k (denominator min(k, |truth|), empty truth -> 0)."""
    n = len(top_idx)
    rel = truth[np.arange(n)[:, None], top_idx[:, :k]].astype(np.float64)
    prec = np.cumsum(rel, axis=1) / np.arange(1, k + 1)
    denom = np.minimum(k, truth.sum(axis=1))
    ap = (prec * rel).sum(axis=1) / np.maximum(denom, 1)
    ap[denom == 0] = 0.0
    return ap


def run_tests():
    cases = [
        ("perfect list", [1, 2, 3, 4, 5], {1, 2, 3, 4, 5}, 1.0),
        ("hits at k=2 and k=4, |truth|=2", [9, 1, 8, 2, 7], {1, 2}, (1 / 2 + 2 / 4) / 2),
        ("single hit at k=5, |truth|=6 -> denominator 5", [1, 2, 3, 4, 5], {5, 6, 7, 8, 9, 10}, (1 / 5) / 5),
        ("empty truth", [1, 2, 3, 4, 5], set(), 0.0),
    ]
    out = []
    for name, pred, truth, expected in cases:
        got = ap_at_k(pred, truth)
        ok = abs(got - expected) < 1e-12
        print(f"[{'OK' if ok else 'FAIL'}] {name}: pred={pred} truth={sorted(truth)} rel={rel_vector(pred, truth)} "
              f"AP@5={got:.6f} expected={expected:.6f}")
        assert ok
        out.append({"case": name, "pred": pred, "truth": sorted(truth), "ap5": got, "expected": expected, "ok": ok})
    return out


if __name__ == "__main__":
    run_tests()
