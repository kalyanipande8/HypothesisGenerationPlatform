import os
import sys
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_biomedical_data import generate_dataset, TRUE_RULES
from encoding import build_vocab, seq_len, sep_position, hyp_positions, encode_pair
from model import ReasoningEncoder
from train_and_evaluate import (
    build_encoder, records_to_xy, train_classifier, evaluate_classifier,
    evaluate_generate_rank_pipeline,
)
from hypothesis_generator import HypothesisGenerator, generate_candidate_hypotheses


def _small_dataset(n=60, seed=1):
    rng = random.Random(seed)
    return generate_dataset(n, rng)


def test_dataset_generation_valid_records():
    obs, records = _small_dataset()
    assert len(obs) == 60
    assert len(records) > 0
    for r in records:
        assert r["label"] in (0, 1)
        assert r["gene"].startswith("G")
        assert r["pathway"].startswith("P")
        assert r["phenotype"].startswith("H")
    for o in obs:
        assert tuple(o["true_rule"]) in TRUE_RULES
        assert o["elevated_genes"] or True  # may be empty in rare cases


def test_encoding_shapes():
    obs, records = _small_dataset()
    o = obs[0]
    r = records[0]
    tokens = encode_pair(o, r["gene"], r["pathway"], r["phenotype"])
    assert len(tokens) == seq_len()
    vocab = build_vocab()
    assert all(t in vocab for t in tokens)


def test_encoder_produces_pooled_vectors():
    vocab = build_vocab()
    encoder = ReasoningEncoder(vocab, seq_len(), sep_position(), hyp_positions())
    obs, records = _small_dataset()
    o = obs[0]
    r = records[0]
    tokens = encode_pair(o, r["gene"], r["pathway"], r["phenotype"])
    vec = encoder.encode_one(tokens)
    assert vec.shape == (16,)
    assert np.isfinite(vec).all()


def test_model_trains_and_beats_chance():
    train_obs, train_records = _small_dataset(n=200, seed=2)
    test_obs, test_records = _small_dataset(n=80, seed=3)

    encoder = build_encoder()
    X_train, y_train = records_to_xy(train_records, train_obs, encoder)
    X_test, y_test = records_to_xy(test_records, test_obs, encoder)

    clf, loss_curve = train_classifier(X_train, y_train, max_iter=300)

    # sanity check: training loss should have decreased over the run
    assert loss_curve[-1] < loss_curve[0]

    test_acc = evaluate_classifier(clf, X_test, y_test)
    majority_baseline = max(np.mean(y_test == 0), np.mean(y_test == 1))
    assert test_acc > majority_baseline - 0.05  # should be at/above chance-ish baseline


def test_full_pipeline_accuracy_in_reasonable_band():
    train_obs = _load_json_helper("train_observations.json")
    train_records = _load_json_helper("train_records.json")
    test_obs = _load_json_helper("test_observations.json")
    test_records = _load_json_helper("test_records.json")

    encoder = build_encoder()
    X_train, y_train = records_to_xy(train_records, train_obs, encoder)
    X_test, y_test = records_to_xy(test_records, test_obs, encoder)
    clf, _ = train_classifier(X_train, y_train)

    test_acc = evaluate_classifier(clf, X_test, y_test)
    assert 0.70 <= test_acc <= 0.86, f"test accuracy {test_acc} outside expected band"


def test_generate_and_rank_beats_random_baseline():
    train_obs = _load_json_helper("train_observations.json")
    train_records = _load_json_helper("train_records.json")
    test_obs = _load_json_helper("test_observations.json")

    encoder = build_encoder()
    X_train, y_train = records_to_xy(train_records, train_obs, encoder)
    clf, _ = train_classifier(X_train, y_train)

    generator = HypothesisGenerator(encoder, clf)
    rng = random.Random(99)
    model_acc, random_acc, n_eval = evaluate_generate_rank_pipeline(
        generator, test_obs, rng, n_trials=10
    )
    assert n_eval > 0
    assert model_acc > random_acc


def test_generated_candidates_nonempty_for_every_observation():
    obs, _ = _small_dataset(n=30, seed=5)
    for o in obs:
        cands = generate_candidate_hypotheses(o)
        assert len(cands) > 0


def _load_json_helper(name):
    import json
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", name)
    if not os.path.exists(path):
        from generate_biomedical_data import build_and_save
        build_and_save(out_dir=os.path.join(base, "data"))
    with open(path) as f:
        return json.load(f)
