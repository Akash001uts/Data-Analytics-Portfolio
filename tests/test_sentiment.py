"""Sentiment pipeline checks on synthetic reviews (no real review text). Needs no data."""

import gzip

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from dap.sentiment import clean as C
from dap.sentiment import evaluate as E
from dap.sentiment import models as M
from dap.sentiment.readme import difference


def block(pid, uid, score, text, name=b"Reviewer", summary=b"Summary"):
    return (
        b"product/productId: " + pid + b"\n"
        b"review/userId: " + uid + b"\n"
        b"review/profileName: " + name + b"\n"
        b"review/helpfulness: 0/0\n"
        b"review/score: " + score + b".0\n"
        b"review/time: 1300000000\n"
        b"review/summary: " + summary + b"\n"
        b"review/text: " + text + b"\n\n"
    )


@pytest.fixture
def snap_file(tmp_path):
    parts = [
        block(b"P1", b"U1", b"5", b"Great snack.<br />Would buy again &quot;soon&quot;."),
        # the same reviewer and text under a second product: a duplicate
        block(b"P2", b"U1", b"5", b"Great snack.<br />Would buy again &quot;soon&quot;."),
        # a profile name with a line break, and a Windows-1252 apostrophe (0x92)
        block(b"P3", b"U2", b"1", b"Didn\x92t like it at all.", name=b"Jo\nSmith"),
        # the same text from two reviewers who agree: keep one
        block(b"P4", b"U3", b"4", b"Tasty and cheap."),
        block(b"P5", b"U4", b"5", b"tasty  and cheap."),
        # the same text from two reviewers who disagree: drop both
        block(b"P6", b"U5", b"1", b"It is what it is."),
        block(b"P7", b"U6", b"5", b"It is what it is."),
        block(b"P8", b"U7", b"3", "Café blend, fine I guess.".encode()),
    ]
    path = tmp_path / "finefoods.txt.gz"
    with gzip.open(path, "wb") as f:
        f.write(b"".join(parts))
    return path


def test_parse_handles_the_file_quirks(snap_file):
    df = C.parse(snap_file)
    assert len(df) == 8 and list(df.review_id) == list(range(8))
    assert df.loc[2, "profile_name"] == "Jo Smith"
    assert df.loc[2, "text"] == "Didn’t like it at all."
    assert df.loc[7, "text"].startswith("Café")
    assert df.score.dtype.kind == "i" and set(df.columns) >= {"user_id", "text", "summary"}


def test_clean_text():
    assert C.clean_text("a<br />b<BR>c &amp; &quot;d&quot;") == 'a b c & "d"'


def test_labels_from_stars():
    labels = C.to_label(pd.Series([1, 2, 3, 4, 5]))
    assert list(labels) == ["negative", "negative", "neutral", "positive", "positive"]


def test_dedupe_counts_and_rules(snap_file):
    df, counts = C.dedupe(C.prepare(C.parse(snap_file)))
    assert counts["dropped_same_reviewer_same_text"] == 1
    assert counts["dropped_cross_reviewer_conflicting_label"] == 2
    assert counts["dropped_cross_reviewer_same_label"] == 1
    assert counts["after_dedupe"] == len(df) == 4
    assert df.norm.is_unique
    assert "It is what it is." not in set(df.text)
    assert 3 in set(df.review_id) and 4 not in set(df.review_id)  # first copy kept


def synthetic_reviews(n_users=3000, per_user=3, seed=0):
    rng = np.random.default_rng(seed)
    users = np.repeat([f"U{i}" for i in range(n_users)], per_user)
    n = len(users)
    df = pd.DataFrame(
        {
            "review_id": np.arange(n),
            "user_id": users,
            "score": rng.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.05, 0.08, 0.15, 0.62]),
            "text": [f"review number {i}" for i in range(n)],
        }
    )
    df["label"] = C.to_label(df.score)
    df["norm"] = df.text.map(C.normalise)
    return df


def test_no_reviewer_or_text_in_two_splits():
    df = synthetic_reviews()
    df["split"] = C.reviewer_split(df.user_id)
    assert (df.groupby("user_id").split.nunique() == 1).all()
    assert (df.groupby("norm").split.nunique() == 1).all()
    shares = df.drop_duplicates("user_id").split.value_counts(normalize=True)
    for name, pct in C.SPLITS.items():
        assert shares[name] == pytest.approx(pct / 100, abs=0.03)


def test_split_does_not_depend_on_row_order():
    df = synthetic_reviews(n_users=200)
    a = C.reviewer_split(df.user_id)
    b = C.reviewer_split(df.user_id.iloc[::-1]).reindex(a.index)
    pd.testing.assert_series_equal(a, b)


def test_load_dedupes_before_splitting(snap_file):
    df, counts = C.load(snap_file)
    assert counts["after_dedupe"] == len(df)
    assert (df.groupby("norm").split.nunique() == 1).all()
    assert sum(counts[f"{s}_reviews"] for s in C.SPLITS) == len(df)


def test_eval_sample_is_stratified_and_repeatable():
    df = synthetic_reviews()
    df["split"] = C.reviewer_split(df.user_id)
    test = df[df.split == "test"]
    a = C.eval_sample(test, n=500)
    b = C.eval_sample(test.sample(frac=1, random_state=3), n=500)
    assert len(a) == 500 and list(a.review_id) == list(b.review_id)
    assert set(a.review_id) <= set(test.review_id)
    want = test.label.value_counts(normalize=True)
    got = a.label.value_counts(normalize=True)
    assert (got - want).abs().max() < 0.01


def test_tfidf_vocabulary_comes_from_training_text_only():
    train = pd.Series(["good tasty snack", "bad stale snack", "okay snack"] * 5)
    y = pd.Series(["positive", "negative", "neutral"] * 5)
    val = pd.Series(["zzleakword good", "zzleakword bad", "zzleakword okay"])
    yv = pd.Series(["positive", "negative", "neutral"])
    model = M.TfidfLogistic(min_df=1).fit(
        train, y, val, yv, grid=[{"C": 1.0, "class_weight": None}]
    )
    assert not any("zzleakword" in t for t in model.vectoriser.vocabulary_)
    assert list(model.predict(pd.Series(["good tasty"]))) == ["positive"]


def test_vader_threshold_tuning_finds_a_clean_split():
    compound = np.array([-0.8, -0.6, -0.1, 0.0, 0.1, 0.7, 0.9])
    y = np.array(["negative", "negative", "neutral", "neutral", "neutral", "positive", "positive"])
    (lo, hi), f1 = M.tune_vader(compound, y)
    assert f1 == pytest.approx(1.0)
    assert list(M.vader_labels(compound, (lo, hi))) == list(y)


def test_vader_runs():
    out = M.vader_compound(pd.Series(["I love this, it is wonderful", "Awful. I hate it."]))
    assert out[0] > 0.5 and out[1] < -0.5


def test_roberta_labels_take_the_most_likely_class():
    probs = pd.DataFrame(
        {"p_negative": [0.7, 0.1, 0.2], "p_neutral": [0.2, 0.8, 0.2], "p_positive": [0.1, 0.1, 0.6]}
    )
    assert list(M.roberta_labels(probs)) == ["negative", "neutral", "positive"]


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.choice(C.LABELS, 500)
    p = rng.choice(C.LABELS, 500)
    fast = E.macro_f1_codes(E._codes(y), E._codes(p))
    assert fast == pytest.approx(f1_score(y, p, average="macro", labels=list(C.LABELS)))


def test_paired_bootstrap():
    rng = np.random.default_rng(1)
    y = rng.choice(C.LABELS, 400)
    noisy = np.where(rng.random(400) < 0.3, rng.choice(C.LABELS, 400), y)
    single, diffs = E.paired_bootstrap(y, {"perfect": y, "noisy": noisy, "copy": noisy}, n_boot=300)
    assert single["perfect"]["estimate"] == 1.0 and single["perfect"]["ci_low"] == 1.0
    assert single["noisy"]["ci_low"] < single["noisy"]["estimate"] < single["noisy"]["ci_high"]
    assert diffs["perfect minus noisy"]["ci_low"] > 0
    assert diffs["noisy minus copy"] == {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_slices_cover_every_review():
    y = np.array(["positive", "negative", "neutral", "positive"])
    df = pd.DataFrame(
        {
            "text": ["short but sweet", "word " * 120, "fine, three Stars", "great " * 300],
            "label": y,
            "n_tokens": [5, 130, 2, 600],
        }
    )
    out = E.slices(df, {"m": y})
    for group in out.values():
        assert sum(v["n"] for v in group.values()) == len(df)
    assert out["contrast_word"]["has a contrast word"]["n"] == 1
    assert out["truncated_by_roberta"]["truncated"]["n"] == 1
    assert out["cut_for_finetuned"]["cut"]["n"] == 2
    assert out["mentions_stars"]["mentions stars"]["n"] == 1  # "starstruck" wouldn't count


def test_readme_difference_flips_the_sign():
    diffs = {"a minus b": {"estimate": 0.1, "ci_low": 0.05, "ci_high": 0.2}}
    assert difference(diffs, "b", "a") == {"estimate": -0.1, "ci_low": -0.2, "ci_high": -0.05}


def test_missing_roberta_cache_says_what_to_run(tmp_path):
    from dap.sentiment.transformer import read_cache

    with pytest.raises(FileNotFoundError, match="dap sentiment transformer"):
        read_cache(tmp_path / "nope.csv")


def test_finetuning_data_comes_from_train_and_validation_reviewers_only(monkeypatch):
    from dap.sentiment import finetune as F

    monkeypatch.setattr(F, "TRAIN_SIZE", 1000)
    monkeypatch.setattr(F, "VAL_SIZE", 200)
    df = synthetic_reviews()
    df["split"] = C.reviewer_split(df.user_id)
    df["in_eval"] = df.review_id.isin(C.eval_sample(df[df.split == "test"], n=500).review_id)
    train, val = F.training_data(df)
    assert len(train) == 1000 and len(val) == 200
    assert set(train.split) == {"train"} and set(val.split) == {"validation"}
    assert not (train.in_eval.any() or val.in_eval.any())
    assert not set(train.user_id) & set(val.user_id)
    want = df[df.split == "train"].label.value_counts(normalize=True)
    assert (train.label.value_counts(normalize=True) - want).abs().max() < 0.01
    again, _ = F.training_data(df.sample(frac=1, random_state=5))
    assert list(again.review_id) == list(train.review_id)


def test_finetuning_class_weights_are_balanced():
    from dap.sentiment.finetune import class_weights

    labels = ["positive"] * 8 + ["negative"] * 2 + ["neutral"] * 2
    counts = np.array([2, 2, 8])  # in LABELS order
    assert np.allclose(class_weights(labels) * counts, len(labels) / 3)


def test_length_batches_use_every_review_once():
    from dap.sentiment.finetune import length_batches

    lengths = np.random.default_rng(1).integers(5, 300, 1003)
    batches = length_batches(lengths, 16, seed=0)
    assert sorted(np.concatenate(batches)) == list(range(1003))
    assert max(len(b) for b in batches) == 16
    again = length_batches(lengths, 16, seed=0)
    assert all((a == b).all() for a, b in zip(batches, again, strict=True))


def test_finetuned_labels_read_their_own_columns():
    probs = pd.DataFrame(
        {
            "p_negative": [0.9],
            "p_neutral": [0.05],
            "p_positive": [0.05],
            "ft_p_negative": [0.1],
            "ft_p_neutral": [0.2],
            "ft_p_positive": [0.7],
        }
    )
    assert list(M.roberta_labels(probs, prefix="ft_p_")) == ["positive"]


def test_missing_finetuned_cache_says_what_to_run(tmp_path):
    from dap.sentiment.finetune import read_cache

    with pytest.raises(FileNotFoundError, match="dap sentiment finetune"):
        read_cache(tmp_path / "nope.csv")
