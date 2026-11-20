"""Testes das funções de src/evaluate.py."""

import numpy as np

from src.evaluate import (
    TrainIndex,
    crop_bars,
    diversity_report,
    longest_run,
    memorization_report,
    ngram_hashes,
    overlap_and_kl,
    two_prop_p,
    wilson,
)
from src.sample import SpecialIds

SP = SpecialIds(bos=1, eos=2, pad=0, null=3, emotions=[4, 5, 6, 7])
BAR = 8


def test_crop_bars():
    seq = [1, 4, BAR, 10, 11, BAR, 12, BAR, 13, 2]
    assert crop_bars(seq, 2, BAR, SP) == [1, 4, BAR, 10, 11, BAR, 12, 2]
    assert crop_bars(seq, 0, BAR, SP) == seq
    assert crop_bars(seq, 9, BAR, SP) == seq


def test_ngram_hashes_detect_equal_ngrams():
    h = ngram_hashes([5, 6, 7, 5, 6, 7], 3)
    assert len(h) == 4 and h[0] == h[3] and len(set(h.tolist())) == 3
    assert len(ngram_hashes([1, 2], 3)) == 0


def test_overlap_and_kl():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    oa, kl = overlap_and_kl(a, b)
    assert oa > 0.8 and kl < 0.3  # ruído da KDE nas caudas
    oa_far, kl_far = overlap_and_kl(a, b + 5)
    assert oa_far < 0.05 and kl_far > kl
    assert np.isfinite(overlap_and_kl(np.ones(10), np.ones(10))[0])  # valores constantes


def test_memorization_counts_copies():
    train = [[1, 4, *range(10, 40), 2]]
    idx = TrainIndex(train, SP, n=4)
    copy = [1, 5, *range(10, 30), 2]
    novel = [1, 5, *range(100, 120), 2]
    rep = memorization_report([copy, novel], SP, idx)
    assert rep["copied_frac"] == 0.5 and rep["longest_copy_max"] == 20
    assert longest_run(np.array([1, 1, 0, 1, 1, 1], bool)) == 3


def test_diversity_identical_is_zero():
    seqs = [[1, 4, *range(10, 30), 2]] * 3
    rep = diversity_report(seqs, ["Q1"] * 3, [[1 / 12] * 12] * 3, SP)
    assert rep["pch_distance"] == 0 and rep["ngram_distance"] == 0


def test_stats_helpers():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    assert two_prop_p(50, 100, 50, 100) == 1.0
    assert two_prop_p(80, 100, 20, 100) < 1e-6


def test_estimate_mode():
    from src.evaluate import KK_MAJOR, KK_MINOR, estimate_mode

    for k in range(12):
        assert estimate_mode(np.roll(KK_MAJOR, k)) == 0
        assert estimate_mode(np.roll(KK_MINOR, k)) == 1
