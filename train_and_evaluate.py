"""
train_and_evaluate.py -- trains a classifier head on top of the fixed
ReasoningEncoder's pooled attention features to predict synthetic-expert
plausibility labels, evaluates held-out accuracy, and separately evaluates
the full generate-then-rank pipeline against a random-generation baseline.
"""
import json
import os

import numpy as np
from sklearn.neural_network import MLPClassifier

from encoding import build_vocab, seq_len, sep_position, hyp_positions, encode_pair
from model import ReasoningEncoder
from hypothesis_generator import HypothesisGenerator, generate_candidate_hypotheses

DATA_DIR = "data"
RNG_SEED = 42


def _load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def build_encoder():
    vocab = build_vocab()
    return ReasoningEncoder(
        vocab=vocab,
        seq_len=seq_len(),
        sep_pos=sep_position(),
        hyp_positions=hyp_positions(),
    )


def records_to_xy(records, observations, encoder):
    obs_by_id = {o["id"]: o for o in observations}
    seqs = []
    y = []
    for r in records:
        obs = obs_by_id[r["obs_id"]]
        seqs.append(encode_pair(obs, r["gene"], r["pathway"], r["phenotype"]))
        y.append(r["label"])
    X = encoder.encode_batch(seqs)
    return X, np.array(y)


def train_classifier(X_train, y_train, hidden=(24,), max_iter=400, seed=RNG_SEED):
    """Trains the classifier incrementally so we can record a real training
    curve (loss per iteration) for the report figure, and returns both the
    fitted classifier and that curve."""
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        max_iter=max_iter,
        random_state=seed,
        early_stopping=False,
        learning_rate_init=0.01,
    )
    clf.fit(X_train, y_train)
    return clf, clf.loss_curve_


def evaluate_classifier(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = float(np.mean(preds == y_test))
    return acc


def evaluate_generate_rank_pipeline(generator, test_observations, rng, n_trials=1):
    """For each held-out test observation: does the top-ranked GENERATED
    hypothesis match the true latent causal rule? Compare the model-ranked
    top-1 hit rate against a RANDOM baseline that picks a random order over
    the same generated candidate set (averaged over n_trials random
    orderings per observation for a stable estimate)."""
    model_hits = 0
    random_hits = 0
    total = 0
    for obs in test_observations:
        candidates = generate_candidate_hypotheses(obs)
        if not candidates:
            continue
        scores = generator.score_candidates(obs, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda t: -t[1])
        top1 = ranked[0][0]
        model_hits += int(tuple(top1) == tuple(obs["true_rule"]))

        for _ in range(n_trials):
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            random_hits += int(tuple(shuffled[0]) == tuple(obs["true_rule"]))
        total += 1

    model_acc = model_hits / total if total else 0.0
    random_acc = random_hits / (total * n_trials) if total else 0.0
    return model_acc, random_acc, total


def main():
    train_obs = _load_json("train_observations.json")
    train_records = _load_json("train_records.json")
    test_obs = _load_json("test_observations.json")
    test_records = _load_json("test_records.json")

    encoder = build_encoder()

    X_train, y_train = records_to_xy(train_records, train_obs, encoder)
    X_test, y_test = records_to_xy(test_records, test_obs, encoder)

    clf, loss_curve = train_classifier(X_train, y_train)

    train_acc = evaluate_classifier(clf, X_train, y_train)
    test_acc = evaluate_classifier(clf, X_test, y_test)

    print(f"train accuracy vs synthetic expert labels: {train_acc:.4f}")
    print(f"test  accuracy vs synthetic expert labels: {test_acc:.4f}")

    generator = HypothesisGenerator(encoder, clf)
    import random as _random
    rng = _random.Random(RNG_SEED)
    model_acc, random_acc, n_eval = evaluate_generate_rank_pipeline(
        generator, test_obs, rng, n_trials=20
    )
    print(f"generate+rank top-1 match rate (model):  {model_acc:.4f} over {n_eval} test observations")
    print(f"generate+rank top-1 match rate (random): {random_acc:.4f} baseline")

    results = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "loss_curve": loss_curve,
        "generate_rank_model_acc": model_acc,
        "generate_rank_random_acc": random_acc,
        "n_eval_observations": n_eval,
    }
    os.makedirs("output", exist_ok=True)
    with open("output/results.json", "w") as f:
        json.dump(results, f, indent=1)

    return encoder, clf, results


if __name__ == "__main__":
    main()
