"""
generate_biomedical_data.py -- synthetic biomedical "observation + candidate
hypothesis" dataset with a small set of hidden ground-truth causal rules
standing in for real biological mechanisms, and a synthetic "expert
plausibility label" standing in for real expert predictions.

No real biomedical database of any kind is used. Everything here is
generated from fixed random seeds so the whole pipeline is reproducible.

LATENT WORLD
------------
- NUM_GENES genes, NUM_PATHWAYS pathways, NUM_PHENOTYPES phenotypes.
- A small fixed set of TRUE_RULES, each a triple (gene, pathway, phenotype)
  meaning "gene regulates pathway, producing phenotype". This is the hidden
  causal structure that a real expert would (imperfectly) know and that the
  hypothesis-generation platform is trying to recover.

OBSERVATION RECORD
-------------------
Each observation simulates one experimental readout:
  - one TRUE_RULE is sampled as the "active" rule driving this observation
  - that rule's gene gets an elevated expression bucket, its pathway gets an
    elevated activity bucket, and its phenotype is flagged present
  - a handful of OTHER genes/pathways/phenotypes get randomly sampled
    (mostly low) levels as background noise / distractors
  - the observation record stores which genes/pathways/phenotypes are
    "elevated" (bucket >= ELEVATED_THRESHOLD) -- this elevated set is what
    the hypothesis generator will combinatorially reason over.

CANDIDATE HYPOTHESIS + LABEL
------------------------------
For each observation we enumerate template hypotheses
  "gene G_i regulates pathway P_j under phenotype H_k"
over the elevated entities in that observation (plus a few distractor
triples drawn from non-elevated entities, so the classifier also sees
genuinely implausible candidates). Each candidate gets a binary
"expert-labeled" plausibility target:
  - label = 1 if the triple exactly matches the observation's true active
    rule (a real mechanistic match)
  - label = 0 otherwise
then REALISTIC LABEL NOISE is injected (flip probability LABEL_NOISE) to
mimic imperfect expert judgement -- this keeps the ceiling on achievable
accuracy below 100%, which is the honest reason the target accuracy in
this project is ~78%, not 100%.
"""
import json
import random

SEED = 13
NUM_GENES = 8
NUM_PATHWAYS = 4
NUM_PHENOTYPES = 3
ELEVATED_THRESHOLD = 2  # bucket values are 0,1,2,3; >=2 counts as "elevated"
LABEL_NOISE = 0.12
N_DISTRACTOR_ENTITIES = 2  # extra random elevated distractor genes/pathways per obs
N_TRAIN_OBS = 500
N_TEST_OBS = 150

GENES = [f"G{i}" for i in range(NUM_GENES)]
PATHWAYS = [f"P{i}" for i in range(NUM_PATHWAYS)]
PHENOTYPES = [f"H{i}" for i in range(NUM_PHENOTYPES)]

# Fixed hidden causal structure: 6 true (gene, pathway, phenotype) rules.
TRUE_RULES = [
    ("G0", "P0", "H0"),
    ("G1", "P1", "H1"),
    ("G2", "P0", "H2"),
    ("G3", "P2", "H0"),
    ("G5", "P3", "H1"),
    ("G6", "P2", "H2"),
]


def _make_observation(rng, obs_id, rule):
    gene, pathway, pheno = rule
    gene_level = {g: rng.choice([0, 0, 1], ) for g in GENES}
    pathway_level = {p: rng.choice([0, 0, 1]) for p in PATHWAYS}
    pheno_present = {h: 0 for h in PHENOTYPES}

    gene_level[gene] = rng.choice([2, 3])
    pathway_level[pathway] = rng.choice([2, 3])
    pheno_present[pheno] = 1

    # distractor elevated entities unrelated to the active rule
    other_genes = [g for g in GENES if g != gene]
    other_pathways = [p for p in PATHWAYS if p != pathway]
    for g in rng.sample(other_genes, k=min(N_DISTRACTOR_ENTITIES, len(other_genes))):
        if rng.random() < 0.5:
            gene_level[g] = rng.choice([2, 3])
    for p in rng.sample(other_pathways, k=min(1, len(other_pathways))):
        if rng.random() < 0.3:
            pathway_level[p] = rng.choice([2, 3])

    elevated_genes = [g for g in GENES if gene_level[g] >= ELEVATED_THRESHOLD]
    elevated_pathways = [p for p in PATHWAYS if pathway_level[p] >= ELEVATED_THRESHOLD]
    present_phenos = [h for h in PHENOTYPES if pheno_present[h] == 1] or [pheno]

    return {
        "id": obs_id,
        "gene_level": gene_level,
        "pathway_level": pathway_level,
        "pheno_present": pheno_present,
        "elevated_genes": elevated_genes,
        "elevated_pathways": elevated_pathways,
        "present_phenotypes": present_phenos,
        "true_rule": list(rule),
    }


def _candidates_for_observation(rng, obs):
    """Combinatorial template grammar over an observation's entities.

    Enumerates (gene, pathway, phenotype) triples from the elevated
    genes/pathways and present phenotypes of this observation (the
    "plausible search space" a real hypothesis generator would consider),
    plus a few fully-random distractor triples over non-elevated entities
    for contrast/negative examples.
    """
    cands = []
    genes = obs["elevated_genes"] or GENES[:1]
    pathways = obs["elevated_pathways"] or PATHWAYS[:1]
    phenos = obs["present_phenotypes"] or PHENOTYPES[:1]
    for g in genes:
        for p in pathways:
            for h in phenos:
                cands.append((g, p, h))

    # a few distractor triples drawn from entities NOT elevated in this obs
    low_genes = [g for g in GENES if g not in genes]
    low_pathways = [p for p in PATHWAYS if p not in pathways]
    for _ in range(3):
        if low_genes and low_pathways:
            g = rng.choice(low_genes)
            p = rng.choice(low_pathways)
            h = rng.choice(PHENOTYPES)
            cands.append((g, p, h))

    # de-duplicate
    seen = set()
    uniq = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _label_candidate(rng, obs, cand):
    true_match = tuple(cand) == tuple(obs["true_rule"])
    label = 1 if true_match else 0
    if rng.random() < LABEL_NOISE:
        label = 1 - label
    return label


def generate_dataset(n_obs, rng):
    observations = []
    records = []
    for i in range(n_obs):
        rule = rng.choice(TRUE_RULES)
        obs = _make_observation(rng, i, rule)
        observations.append(obs)
        for cand in _candidates_for_observation(rng, obs):
            label = _label_candidate(rng, obs, cand)
            records.append({
                "obs_id": obs["id"],
                "gene": cand[0],
                "pathway": cand[1],
                "phenotype": cand[2],
                "is_true_match": int(tuple(cand) == tuple(obs["true_rule"])),
                "label": label,
            })
    return observations, records


def build_and_save(out_dir="data"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(SEED)

    train_obs, train_records = generate_dataset(N_TRAIN_OBS, rng)
    test_obs, test_records = generate_dataset(N_TEST_OBS, rng)

    with open(os.path.join(out_dir, "train_observations.json"), "w") as f:
        json.dump(train_obs, f, indent=1)
    with open(os.path.join(out_dir, "train_records.json"), "w") as f:
        json.dump(train_records, f, indent=1)
    with open(os.path.join(out_dir, "test_observations.json"), "w") as f:
        json.dump(test_obs, f, indent=1)
    with open(os.path.join(out_dir, "test_records.json"), "w") as f:
        json.dump(test_records, f, indent=1)

    print(f"train: {len(train_obs)} observations, {len(train_records)} labeled hypothesis records")
    print(f"test:  {len(test_obs)} observations, {len(test_records)} labeled hypothesis records")
    pos_rate_train = sum(r["label"] for r in train_records) / len(train_records)
    pos_rate_test = sum(r["label"] for r in test_records) / len(test_records)
    print(f"positive-label rate: train={pos_rate_train:.3f} test={pos_rate_test:.3f}")
    return train_obs, train_records, test_obs, test_records


if __name__ == "__main__":
    build_and_save()
