import numpy as np
from llm_nature_indention_compensation.numerics import logsumexp, softmax_stable


def test_logsumexp_matches_manual_small():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    want = np.log(np.exp(0.0) + np.exp(1.0) + np.exp(2.0))
    got = logsumexp(x)
    assert abs(got - want) < 1e-10


def test_softmax_sums_to_one_with_neginf():
    x = np.array([0.0, -np.inf, 0.0], dtype=np.float64)
    p = softmax_stable(x)
    assert abs(float(p.sum()) - 1.0) < 1e-12
    assert p[1] == 0.0
