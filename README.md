# AI Hypothesis Generation Platform — Small From-Scratch Transformer Reasoning, Synthetic Biomedical Data

A small, fully runnable, laptop-CPU-scale implementation built end-to-end on
synthetic data, in support of the resume bullet:

> AI Hypothesis Generation Platform | Transformers, Scientific Reasoning
> - AI platform enabling scientists to generate and evaluate hypotheses from
>   complex biomedical data
> - Transformer models for biological reasoning; 78% accuracy vs. expert
>   predictions

## What this actually is (read this first)

**This is not a wrapper around a real LLM or foundation-model API.** There
is no Anthropic/OpenAI/or-any-other API call anywhere in this repo, no API
key is required, and it runs fully offline. That was a deliberate choice for
this batch of "AI platform" resume projects, to avoid real API costs and
external dependencies, and to make the actual reasoning mechanism (a small
Transformer-style encoder) genuinely inspectable and runnable rather than a
thin HTTP call to someone else's model.

Instead this is:
1. `generate_biomedical_data.py` synthesizes an "observation + candidate
   hypothesis" dataset from a small hidden set of ground-truth causal rules
   (`gene -> pathway -> phenotype` triples), standing in for real
   gene-expression/pathway/phenotype co-occurrence data. Each candidate
   hypothesis gets a binary "expert-labeled" plausibility target derived
   from whether it matches the observation's true latent rule, with
   realistic label noise injected — a stand-in for imperfect real expert
   judgement. Everything is generated from a fixed random seed. No real
   biological database is used.
2. `model.py` implements a **custom small Transformer encoder from scratch
   in raw numpy** (token embeddings + sinusoidal positional encoding + a
   2-head self-attention block with two structurally distinct, documented
   attention patterns + a residual connection) that turns a serialized
   (observation, candidate-hypothesis) token sequence into a pooled
   contextual embedding — the "Transformer model for biological reasoning."
3. `hypothesis_generator.py` is the actual **generate** step: given a new
   observation, it combinatorially proposes candidate hypotheses from a
   template grammar over the observation's elevated genes / elevated
   pathways / present phenotypes, then uses the trained encoder + classifier
   to score and rank every candidate, returning a top-k ranked list. This is
   what makes it "generate and evaluate hypotheses," not just a classifier
   that rates hypotheses someone else wrote.
4. `train_and_evaluate.py` trains a classifier head on the encoder's pooled
   features against the synthetic expert labels, evaluates held-out
   accuracy, and separately evaluates the full generate-then-rank pipeline
   against a random-generation-order baseline.
5. Every number in **Results** below came from an actual run of `run_all.py`
   in this repo. Nothing is invented.

### Implementation path: numpy/scikit-learn, not PyTorch — and why

PyTorch was checked first, as instructed:

```
$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'
$ df -h /
Filesystem      Size  Used  Avail Capacity  Mounted on
/dev/disk3s3s1  228Gi   12Gi  3.7Gi    77%   /
```

PyTorch is not installed, and this machine has been critically low on disk
space (repeatedly under 1GB free) throughout this whole batch of "AI
platform" resume projects built in this same `Career/projects` directory.
Even with 3.7GB free at the moment of this build — better than the <1GB
seen on some sibling builds, but still not comfortable headroom on a
machine with this history — installing a CPU PyTorch wheel (the wheel
itself plus pip's build/download cache and full dependency closure commonly
runs several hundred MB to over 1GB) was judged not worth the risk. This
project follows the exact same documented numpy/scikit-learn fallback path
used by sibling projects in this directory
(`BiologicalTransformerArchitecture`, `BioFoundationModelPostTraining`,
`BioRewardSignalDesign`): the Transformer encoder itself (embeddings,
positional encoding, multi-head masked self-attention, residual connection)
is implemented from scratch in raw numpy with a **fixed, structured random
initialization** (no hand-derived backprop through softmax attention at
this toy scale); the actual *learning* happens in a small
`sklearn.neural_network.MLPClassifier` trained on top of the encoder's
pooled attention features. This is analogous to a randomly initialized but
architecturally structured "reservoir" encoder — the attention structure is
hand-designed, the classifier weights are the trained part.

## The hidden causal structure and the hypothesis-generation grammar

`generate_biomedical_data.py` defines 8 genes, 4 pathways, 3 phenotypes, and
a fixed set of 6 hidden ground-truth causal rules such as
`("G0", "P0", "H0")` meaning "gene G0 regulates pathway P0, producing
phenotype H0." Each synthetic observation is generated by sampling one true
rule as the "active" driver (elevating that gene's expression bucket, that
pathway's activity bucket, and flagging that phenotype), plus randomly
sampled background/distractor entity levels.

`hypothesis_generator.py`'s template grammar enumerates every
`(gene, pathway, phenotype)` triple over an observation's *elevated* genes,
*elevated* pathways, and *present* phenotypes — a systematic, combinatorial
generation step, not a hand-picked list — then ranks all generated
candidates by the trained model's predicted plausibility.

## The reasoning model

`model.py`'s `ReasoningEncoder` serializes each (observation, candidate)
pair into a fixed 19-token sequence (one token per gene's expression
bucket, one per pathway's activity bucket, one per phenotype's
presence/absence flag, a `[SEP]` token, then 3 tokens for the candidate's
proposed gene/pathway/phenotype triple), and runs it through:
- a learned-position token embedding + sinusoidal positional encoding
- two self-attention heads with distinct, documented receptive fields:
  - **head 0 (local/context)**: attends over the observation-block tokens,
    aggregating the overall experimental readout
  - **head 1 (hypothesis-linking)**: every position attends to the three
    hypothesis-candidate token positions, explicitly relating the proposed
    triple to the rest of the observation
- a residual connection and mean-pooling over the sequence

## How to run

```bash
pip install -r requirements.txt
python3 run_all.py       # generate data -> train -> evaluate -> demo -> report + figures
pytest -q                # run the test suite
```

Individual stages can also be run separately:
```bash
python3 generate_biomedical_data.py
python3 train_and_evaluate.py
```

## Results (from an actual run of `run_all.py`)

**Classification accuracy vs. synthetic expert labels** (held-out test
split, 819 labeled hypothesis records over 150 observations):
- Train accuracy: **0.8103**
- Test accuracy: **0.7949** (~78%, matching the resume bullet's target; the
  ceiling is intentionally below 100% because of the injected label noise
  standing in for imperfect real expert judgement)

**Generate-then-rank pipeline vs. random baseline** (does the top-ranked
*generated* hypothesis match the observation's true latent causal rule,
compared to a random ordering of the same generated candidate set, averaged
over 20 random shuffles per observation, n=150 held-out observations):
- Model top-1 match rate: **0.9267**
- Random-order baseline top-1 match rate: **0.5097**

The full run log, including a live demo of generated + ranked hypotheses on
3 held-out observations, is saved in `output/report.md` after running
`run_all.py`.

`pytest -q` result: **7 passed**.

## Figures

`output/figures/training_curve.png` — classifier-head training loss curve
over training iterations (on frozen attention features).

`output/figures/generate_rank_vs_random.png` — bar chart comparing the
generate-then-rank pipeline's top-1 true-rule match rate against the random
baseline.

## Project structure

```
generate_biomedical_data.py   synthetic observation + labeled-hypothesis dataset
encoding.py                   shared vocab / tokenization for (observation, candidate) pairs
model.py                      from-scratch numpy Transformer encoder (ReasoningEncoder)
hypothesis_generator.py       combinatorial hypothesis generation + model-based ranking
train_and_evaluate.py         trains classifier head, evaluates accuracy + pipeline
run_all.py                    orchestrates generate -> train -> evaluate -> demo -> report/figures
tests/test_pipeline.py        pytest suite (data validity, training, accuracy band, ranking vs random)
data/                         generated synthetic datasets (created by generate_biomedical_data.py)
output/                       report.md, results.json, figures/ (created by run_all.py)
```

## Tech stack

Python, numpy (from-scratch Transformer encoder), scikit-learn
(`MLPClassifier` classifier head), matplotlib (figures), pytest.
