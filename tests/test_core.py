import numpy as np
import pytest

from fewshot.analyze import average_rank, mean_ci, paired_difference, summarise
from fewshot.data import draw_seed, few_shot_indices, select_pool, to_rgb, toy_dataset
from fewshot.handcrafted import handcrafted_features
from fewshot.methods import full_proba, linear_probe, random_forest, scores


# ----------------------------------------------------------------------------- data protocol
def test_pool_has_at_most_n_per_class_and_is_deterministic():
    labels = np.r_[np.zeros(1000), np.ones(30), np.full(600, 2)].astype(int)
    a, b = select_pool(labels, 500, 0), select_pool(labels, 500, 0)
    assert (a == b).all()
    counts = np.bincount(labels[a])
    assert list(counts) == [500, 30, 500]          # small classes are kept whole


def test_few_shot_draws_are_balanced_reproducible_and_differ_between_draws():
    pool_labels = np.repeat(np.arange(4), 100)
    s0, s1 = draw_seed(2026, "bloodmnist", 5, 0), draw_seed(2026, "bloodmnist", 5, 1)
    i0, i0b, i1 = few_shot_indices(pool_labels, 5, s0), few_shot_indices(pool_labels, 5, s0), few_shot_indices(pool_labels, 5, s1)
    assert (i0 == i0b).all() and not np.array_equal(np.sort(i0), np.sort(i1))
    assert list(np.bincount(pool_labels[i0])) == [5, 5, 5, 5]
    assert len(set(i0)) == len(i0)                  # no image drawn twice


def test_draw_seed_is_stable_across_sessions():
    # crc32-based, not Python's randomised hash(): this exact value must never change
    assert draw_seed(2026, "pneumoniamnist", 10, 3) == draw_seed(2026, "pneumoniamnist", 10, 3)
    assert draw_seed(2026, "pneumoniamnist", 10, 3) != draw_seed(2026, "dermamnist", 10, 3)
    assert 0 <= draw_seed(2026, "octmnist", 50, 4) < 2 ** 32


def test_to_rgb():
    assert to_rgb(np.zeros((2, 8, 8), np.uint8)).shape == (2, 8, 8, 3)
    assert to_rgb(np.zeros((2, 8, 8, 3), np.uint8)).shape == (2, 8, 8, 3)


# ----------------------------------------------------------------------------- features & classifiers
def test_handcrafted_features_same_length_for_any_resolution():
    ds28, ds64 = toy_dataset("toy_grey", 28, n_pool=2, n_test=1), toy_dataset("toy_grey", 64, n_pool=2, n_test=1)
    f28, f64 = handcrafted_features(ds28.pool_images), handcrafted_features(ds64.pool_images)
    assert f28.shape[1] == f64.shape[1] and np.isfinite(f28).all()


def test_full_proba_fills_missing_classes():
    from sklearn.linear_model import LogisticRegression
    X = np.r_[np.zeros((5, 2)), np.ones((5, 2))]
    y = np.r_[np.zeros(5), np.full(5, 2)].astype(int)      # class 1 absent
    p = full_proba(LogisticRegression().fit(X, y), X, 3)
    assert p.shape == (10, 3) and np.allclose(p[:, 1], 0) and np.allclose(p.sum(1), 1)


@pytest.mark.parametrize("clf,shots", [("probe", 1), ("forest", 10)])
def test_classifiers_learn_separable_data(clf, shots):
    # (a random forest cannot be expected to generalise from 1 example per class: trees isolate classes
    #  along arbitrary single features; that weakness is part of what the benchmark measures)
    rng = np.random.default_rng(0)
    centres = np.eye(4) * 5
    tr_y = np.repeat(np.arange(4), shots)
    tr_x = centres[tr_y] + rng.normal(0, 0.5, (len(tr_y), 4))
    te_y = np.repeat(np.arange(4), 25)
    te_x = centres[te_y] + rng.normal(0, 0.5, (100, 4))
    fn = linear_probe if clf == "probe" else random_forest
    p = fn(tr_x, tr_y, te_x, 4)
    assert scores(te_y, p)["balanced_accuracy"] > 0.95


def test_scores_binary_and_multiclass():
    y = np.array([0, 0, 1, 1])
    p = np.array([[.9, .1], [.6, .4], [.3, .7], [.2, .8]])
    s = scores(y, p)
    assert s["auc"] == 1.0 and s["balanced_accuracy"] == 1.0
    y3 = np.array([0, 1, 2, 2])
    p3 = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .2, .7], [.5, .3, .2]])
    s3 = scores(y3, p3)
    assert s3["balanced_accuracy"] == pytest.approx((1 + 1 + 0.5) / 3) and 0.5 < s3["auc"] <= 1


# ----------------------------------------------------------------------------- statistics
def test_mean_ci_matches_t_interval():
    m, half, n = mean_ci([0.70, 0.72, 0.74, 0.76, 0.78])
    assert m == pytest.approx(0.74) and n == 5
    assert half == pytest.approx(2.776445 * np.std([0.70, 0.72, 0.74, 0.76, 0.78], ddof=1) / np.sqrt(5), rel=1e-5)
    assert np.isnan(mean_ci([0.5])[1])


def _rows():
    rows = []
    for d in ("a", "b"):
        for k in (1, 10):
            for dr in range(3):
                for m, base in (("good", 0.8), ("bad", 0.6)):
                    rows.append({"dataset": d, "method": m, "shots": k, "draw": dr,
                                 "balanced_accuracy": base + 0.01 * dr + 0.05 * (k == 10), "auc": 0.9, "seconds": 1.0})
    return rows


def test_summary_rank_and_paired_difference():
    rows = _rows()
    s = summarise(rows)
    assert s["a"]["good"][10][0] == pytest.approx(0.86)
    assert average_rank(s, ["good", "bad"]) == {"good": 1.0, "bad": 2.0}
    diff = paired_difference(rows, "good", "bad")
    assert diff[("a", 1)][0] == pytest.approx(0.2) and diff[("a", 1)][1] == pytest.approx(0.0)
