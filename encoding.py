"""
encoding.py -- shared vocabulary and tokenization for turning an
(observation, candidate-hypothesis) pair into the fixed token sequence
consumed by model.ReasoningEncoder.

Sequence layout (fixed length, one token per position):
  [G0_level, G1_level, ..., G7_level,      <- NUM_GENES tokens
   P0_level, P1_level, ..., P3_level,      <- NUM_PATHWAYS tokens
   H0_flag,  H1_flag,  H2_flag,            <- NUM_PHENOTYPES tokens
   SEP,
   HYP_GENE_<g>, HYP_PATHWAY_<p>, HYP_PHENO_<h>]   <- 3 hypothesis tokens
"""
from generate_biomedical_data import GENES, PATHWAYS, PHENOTYPES

SEP = "[SEP]"


def build_vocab():
    vocab = []
    for g in GENES:
        for lvl in range(4):
            vocab.append(f"{g}_L{lvl}")
    for p in PATHWAYS:
        for lvl in range(4):
            vocab.append(f"{p}_L{lvl}")
    for h in PHENOTYPES:
        for flag in (0, 1):
            vocab.append(f"{h}_F{flag}")
    vocab.append(SEP)
    for g in GENES:
        vocab.append(f"HYP_GENE_{g}")
    for p in PATHWAYS:
        vocab.append(f"HYP_PATHWAY_{p}")
    for h in PHENOTYPES:
        vocab.append(f"HYP_PHENO_{h}")
    return vocab


def seq_len():
    return len(GENES) + len(PATHWAYS) + len(PHENOTYPES) + 1 + 3


def sep_position():
    return len(GENES) + len(PATHWAYS) + len(PHENOTYPES)


def hyp_positions():
    base = sep_position() + 1
    return [base, base + 1, base + 2]


def encode_pair(obs, gene, pathway, phenotype):
    """Build the fixed token sequence for one (observation, candidate) pair."""
    tokens = []
    for g in GENES:
        tokens.append(f"{g}_L{obs['gene_level'][g]}")
    for p in PATHWAYS:
        tokens.append(f"{p}_L{obs['pathway_level'][p]}")
    for h in PHENOTYPES:
        tokens.append(f"{h}_F{obs['pheno_present'][h]}")
    tokens.append(SEP)
    tokens.append(f"HYP_GENE_{gene}")
    tokens.append(f"HYP_PATHWAY_{pathway}")
    tokens.append(f"HYP_PHENO_{phenotype}")
    assert len(tokens) == seq_len()
    return tokens
