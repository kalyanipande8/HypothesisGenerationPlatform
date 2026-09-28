# Hypothesis Generation Platform -- Run Report

## Classification accuracy vs. synthetic expert labels

- Train accuracy: 0.8103
- Test accuracy: 0.7949

## Generate-then-rank pipeline vs. random baseline

- Model top-1 match rate: 0.9267 (n=150 held-out observations)
- Random-order baseline top-1 match rate: 0.5097

## Demo: generate + rank on held-out observations

### Observation 0 (true latent rule: ['G0', 'P0', 'H0'])

- 1. score=0.505  Hypothesis: gene G0 regulates pathway P0, producing phenotype H0.  [MATCHES TRUE RULE]

### Observation 1 (true latent rule: ['G0', 'P0', 'H0'])

- 1. score=0.483  Hypothesis: gene G0 regulates pathway P0, producing phenotype H0.  [MATCHES TRUE RULE]
- 2. score=0.169  Hypothesis: gene G7 regulates pathway P0, producing phenotype H0.  

### Observation 2 (true latent rule: ['G1', 'P1', 'H1'])

- 1. score=0.619  Hypothesis: gene G5 regulates pathway P3, producing phenotype H1.  
- 2. score=0.493  Hypothesis: gene G1 regulates pathway P1, producing phenotype H1.  [MATCHES TRUE RULE]
- 3. score=0.430  Hypothesis: gene G1 regulates pathway P3, producing phenotype H1.  
- 4. score=0.363  Hypothesis: gene G5 regulates pathway P1, producing phenotype H1.  

