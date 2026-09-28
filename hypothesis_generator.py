"""
hypothesis_generator.py -- the actual GENERATE step of the platform.

Given a new observation, this module:
  1. GENERATES candidate hypotheses combinatorially from a template grammar
     over the observation's elevated genes / elevated pathways / present
     phenotypes ("gene G_i regulates pathway P_j, producing phenotype H_k").
     This is deliberately a systematic enumeration, not a hand-picked list --
     it is the "generate" half of "generate and evaluate hypotheses".
  2. EVALUATES/RANKS every generated candidate using the trained
     ReasoningEncoder + classifier from model.py / train_and_evaluate.py,
     returning a top-k ranked list with plausibility scores.

This is what makes the system a "platform" rather than a plain classifier:
scoring alone can only rate hypotheses that already exist; this module also
proposes them.
"""
from itertools import product

from generate_biomedical_data import GENES, PATHWAYS, PHENOTYPES
from encoding import encode_pair


def generate_candidate_hypotheses(observation):
    """Template-grammar enumeration of candidate (gene, pathway, phenotype)
    hypotheses over an observation's elevated/present entities.

    Falls back to the full entity sets if an observation happens to have no
    elevated genes/pathways (keeps the generator robust on edge cases).
    """
    genes = observation["elevated_genes"] or GENES
    pathways = observation["elevated_pathways"] or PATHWAYS
    phenos = observation["present_phenotypes"] or PHENOTYPES

    candidates = []
    for g, p, h in product(genes, pathways, phenos):
        candidates.append((g, p, h))
    return candidates


def hypothesis_text(cand):
    g, p, h = cand
    return f"Hypothesis: gene {g} regulates pathway {p}, producing phenotype {h}."


class HypothesisGenerator:
    """Wraps a trained (encoder, classifier) pair to generate + rank
    hypotheses for new observations."""

    def __init__(self, encoder, classifier):
        self.encoder = encoder
        self.classifier = classifier

    def score_candidates(self, observation, candidates):
        feats = self.encoder.encode_batch(
            [encode_pair(observation, g, p, h) for (g, p, h) in candidates]
        )
        probs = self.classifier.predict_proba(feats)[:, 1]
        return probs

    def generate_and_rank(self, observation, top_k=5):
        candidates = generate_candidate_hypotheses(observation)
        scores = self.score_candidates(observation, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda t: -t[1])
        top = ranked[:top_k]
        return [
            {
                "hypothesis": hypothesis_text(c),
                "triple": c,
                "score": float(s),
                "matches_true_rule": tuple(c) == tuple(observation["true_rule"]),
            }
            for c, s in top
        ]
