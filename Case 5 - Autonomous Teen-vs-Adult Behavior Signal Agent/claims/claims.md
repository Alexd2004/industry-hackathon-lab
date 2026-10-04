# Slide and pitch claims for A5's slide pass

Edit before the pass. One claim per line starting with "- ". Then run:

    python -m softsignal.agents.a5_audit

A5 checks every claim against the results files and the risk list in claims/risks.yaml, and writes
results/claims_check.csv. Its verdicts are flags for a person: supported, projected (only a projected row
holds the number), unsupported (no results file holds it), or cannot_check (no number, or no file).

The claims below are a starting set taken from the handovers; replace them with the real slide text.

- SoftSignal catches 92% of teens at a 15% false-teen cap on the 900 held-out accounts.
- At a 15% cap, 17.1% of held-out adults are flagged, and 15.0% on training data.
- At a 15% cap, 225 accounts are sent to verification now, at 99.1% precision.
- The keyword baseline catches 50% of teens at 32% false-teen.
- The loop promotes the stack at round 7 and reaches 88.7% recall at 16.9% false-teen.
- The stack's AUC is 0.955.
- The five-agent crew makes the model more accurate.
- The top 100 accounts in the ranked list are all teens.
