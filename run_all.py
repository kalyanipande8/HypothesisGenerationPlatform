"""
run_all.py -- orchestrates the full pipeline:
  generate data -> train -> evaluate -> demo generate+rank on held-out
  observations -> write output/report.md and output/figures/*.png

Run with: python3 run_all.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_biomedical_data import build_and_save
from train_and_evaluate import (
    build_encoder, records_to_xy, train_classifier, evaluate_classifier,
    evaluate_generate_rank_pipeline, _load_json,
)
from hypothesis_generator import HypothesisGenerator


def make_figures(loss_curve, model_acc, random_acc, out_dir="output/figures"):
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(loss_curve) + 1), loss_curve, color="#3b5bdb")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Training loss")
    ax.set_title("Classifier head training curve\n(on frozen attention features)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Model\n(generate+rank)", "Random\nbaseline"],
                   [model_acc, random_acc], color=["#3b5bdb", "#adb5bd"])
    ax.set_ylabel("Top-1 match rate vs. true latent rule")
    ax.set_ylim(0, 1)
    ax.set_title("Generate-then-rank pipeline vs. random baseline")
    for b, v in zip(bars, [model_acc, random_acc]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "generate_rank_vs_random.png"), dpi=150)
    plt.close(fig)


def write_report(train_acc, test_acc, model_acc, random_acc, n_eval, demo_lines):
    os.makedirs("output", exist_ok=True)
    with open("output/report.md", "w") as f:
        f.write("# Hypothesis Generation Platform -- Run Report\n\n")
        f.write("## Classification accuracy vs. synthetic expert labels\n\n")
        f.write(f"- Train accuracy: {train_acc:.4f}\n")
        f.write(f"- Test accuracy: {test_acc:.4f}\n\n")
        f.write("## Generate-then-rank pipeline vs. random baseline\n\n")
        f.write(f"- Model top-1 match rate: {model_acc:.4f} (n={n_eval} held-out observations)\n")
        f.write(f"- Random-order baseline top-1 match rate: {random_acc:.4f}\n\n")
        f.write("## Demo: generate + rank on held-out observations\n\n")
        for block in demo_lines:
            f.write(block + "\n")


def main():
    print("=== 1. Generating synthetic biomedical dataset ===")
    build_and_save()

    print("\n=== 2. Building encoder + training classifier ===")
    encoder = build_encoder()
    train_obs = _load_json("train_observations.json")
    train_records = _load_json("train_records.json")
    test_obs = _load_json("test_observations.json")
    test_records = _load_json("test_records.json")

    X_train, y_train = records_to_xy(train_records, train_obs, encoder)
    X_test, y_test = records_to_xy(test_records, test_obs, encoder)
    clf, loss_curve = train_classifier(X_train, y_train)

    train_acc = evaluate_classifier(clf, X_train, y_train)
    test_acc = evaluate_classifier(clf, X_test, y_test)
    print(f"train accuracy: {train_acc:.4f}  test accuracy: {test_acc:.4f}")

    print("\n=== 3. Evaluating generate+rank pipeline vs random baseline ===")
    generator = HypothesisGenerator(encoder, clf)
    import random as _random
    rng = _random.Random(42)
    model_acc, random_acc, n_eval = evaluate_generate_rank_pipeline(
        generator, test_obs, rng, n_trials=20
    )
    print(f"model top-1 match: {model_acc:.4f}  random baseline: {random_acc:.4f}  (n={n_eval})")

    print("\n=== 4. Demo: generate + rank hypotheses for a few held-out observations ===")
    demo_lines = []
    for obs in test_obs[:3]:
        header = f"### Observation {obs['id']} (true latent rule: {obs['true_rule']})"
        print(header)
        demo_lines.append(header + "\n")
        ranked = generator.generate_and_rank(obs, top_k=5)
        for rank, item in enumerate(ranked, start=1):
            line = (f"{rank}. score={item['score']:.3f}  "
                    f"{item['hypothesis']}  "
                    f"{'[MATCHES TRUE RULE]' if item['matches_true_rule'] else ''}")
            print("   " + line)
            demo_lines.append("- " + line)
        demo_lines.append("")

    print("\n=== 5. Writing figures + report ===")
    make_figures(loss_curve, model_acc, random_acc)
    write_report(train_acc, test_acc, model_acc, random_acc, n_eval, demo_lines)

    results = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "generate_rank_model_acc": model_acc,
        "generate_rank_random_acc": random_acc,
        "n_eval_observations": n_eval,
    }
    with open("output/results.json", "w") as f:
        json.dump(results, f, indent=1)

    print("\nDone. See output/report.md and output/figures/.")


if __name__ == "__main__":
    main()
