"""
model.py -- small from-scratch Transformer encoder for the (observation,
candidate-hypothesis) reasoning task, implemented in raw numpy.

WHY NUMPY, NOT PYTORCH (checked first, documented honestly)
--------------------------------------------------------------------------
    $ python3 -c "import torch"
    ModuleNotFoundError: No module named 'torch'
    $ df -h /
    ... Avail 3.7Gi ...

PyTorch is not installed in this environment, and this machine has been
critically low on disk space throughout a whole batch of sibling "AI
platform" resume projects built in this same directory
(BiologicalTransformerArchitecture, BioFoundationModelPostTraining,
BioRewardSignalDesign). With well under 5GB free and a documented history of
this box running critically low, installing a CPU torch wheel (the wheel
itself plus pip's build/download cache and dependency closure routinely runs
several hundred MB to >1GB) was judged not safe headroom. This project
follows the exact same documented numpy/scikit-learn fallback path those
sibling projects used.

ARCHITECTURE
--------------------------------------------------------------------------
Each (observation, candidate hypothesis) pair is serialized into a FIXED
token sequence (see encode.py-equivalent logic below / hypothesis_generator
.py) covering:
  - one token per gene, encoding that gene's discretized expression bucket
  - one token per pathway, encoding that pathway's discretized activity bucket
  - one token per phenotype, encoding presence/absence
  - a [SEP] token
  - three tokens for the CANDIDATE hypothesis itself: which gene, which
    pathway, which phenotype it proposes as the causal triple

A custom Transformer encoder (token embedding + sinusoidal positional
encoding + multi-head self-attention with two structurally-distinct heads +
residual connection) turns this token sequence into one pooled contextual
embedding per (observation, hypothesis) pair. This is a genuine, if
lightweight, attention-based encoder -- exactly the "Transformer model for
biological reasoning" component of the resume bullet, not a rebranded bag
of words.

TRAINABLE vs FIXED PARTS (documented simplification, matching sibling
projects in this directory): the token embedding table and the Q/K/V/O
projection matrices are FIXED structured-random init (no hand-derived
backprop through softmax attention at this toy scale). All of the actual
*learning* happens in the small classifier trained on top of the pooled
attention features (see train_and_evaluate.py), analogous to a randomly
initialized but architecturally structured "reservoir" encoder.

Two attention heads, each with an explicit documented job:
  head 0 (local/context head): attends mainly within the observation block
      (genes+pathways+phenotypes), i.e. positions 0..sep_pos.
      Rationale: aggregating the overall experimental readout is a "local
      context" operation over the observation tokens.
  head 1 (hypothesis-linking head): attends mainly from every position to
      the three hypothesis-candidate positions (and itself).
      Rationale: whether a candidate hypothesis is plausible depends on how
      it RELATES to the observation, so this head explicitly links every
      observation token to the proposed gene/pathway/phenotype triple.
"""
import numpy as np

D_MODEL = 16
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS
SEED = 7


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _sinusoidal_positions(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def _build_head_masks(seq_len, sep_pos, hyp_positions):
    idx = np.arange(seq_len)
    i, j = np.meshgrid(idx, idx, indexing="ij")

    # head 0: local/context -- can attend to any observation-block token
    # (<= sep_pos) plus itself; blocks direct attention into hyp tokens
    # unless the query IS a hyp token (so hyp tokens can still see context).
    obs_block = j <= sep_pos
    context_allowed = obs_block | (i == j) | (i > sep_pos)
    head0 = np.where(context_allowed, 0.0, -1e9)

    # head 1: hypothesis-linking -- every query can attend to the 3
    # hypothesis-candidate key positions, plus itself.
    hyp_allowed = np.isin(j, hyp_positions) | (i == j)
    head1 = np.where(hyp_allowed, 0.0, -1e9)

    return [head0, head1]


class ReasoningEncoder:
    """Custom small Transformer encoder over (observation, hypothesis) token
    sequences, with two structurally-distinct attention heads (see module
    docstring)."""

    def __init__(self, vocab, seq_len, sep_pos, hyp_positions,
                 d_model=D_MODEL, n_heads=N_HEADS, seed=SEED):
        assert d_model % n_heads == 0
        self.vocab = vocab
        self.v2i = {tok: idx for idx, tok in enumerate(vocab)}
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.embed = rng.normal(0, scale, size=(len(vocab), d_model))
        self.pos_enc = _sinusoidal_positions(seq_len, d_model)
        self.head_masks = _build_head_masks(seq_len, sep_pos, hyp_positions)

        self.Wq = [rng.normal(0, scale, (d_model, self.d_head)) for _ in range(n_heads)]
        self.Wk = [rng.normal(0, scale, (d_model, self.d_head)) for _ in range(n_heads)]
        self.Wv = [rng.normal(0, scale, (d_model, self.d_head)) for _ in range(n_heads)]
        self.Wo = rng.normal(0, scale, (d_model, d_model))

    def _tokenize(self, tokens):
        idxs = [self.v2i[t] for t in tokens]
        assert len(idxs) == self.seq_len
        return np.array(idxs)

    def encode_one(self, tokens, return_attn=False):
        idxs = self._tokenize(tokens)
        x = self.embed[idxs] + self.pos_enc

        head_outputs = []
        attn_maps = []
        for h in range(self.n_heads):
            Q = x @ self.Wq[h]
            K = x @ self.Wk[h]
            V = x @ self.Wv[h]
            scores = (Q @ K.T) / np.sqrt(self.d_head) + self.head_masks[h]
            attn = _softmax(scores, axis=-1)
            head_outputs.append(attn @ V)
            attn_maps.append(attn)

        concat = np.concatenate(head_outputs, axis=-1)
        out = concat @ self.Wo
        out = out + x  # residual
        pooled = out.mean(axis=0)

        if return_attn:
            return pooled, attn_maps
        return pooled

    def encode_batch(self, seqs):
        return np.stack([self.encode_one(s) for s in seqs], axis=0)


HEAD_NAMES = ["local_context (observation block)", "hypothesis_linking (to candidate triple)"]
