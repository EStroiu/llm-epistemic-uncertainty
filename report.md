# Text Tree Uncertainty Estimation

## How it basically works
It works by looking at the model’s own answer text, not the logits.

In simple terms:

It extracts a few human-readable features from the response, like answer length, reason length, whether the answer is well formatted, whether it uses hedging words, and how specific the reason is.
A small decision tree learns patterns in those features that usually mean the model is wrong.
The tree then outputs an uncertainty score, which we treat as a probability of error.
So instead of asking “how uncertain were the logits?”, it asks “does this answer look weak, vague, or malformed enough that it is likely wrong?”

## Features
They are extracted directly from the generated answer text and the original claim in fever_texttree_uncertainty.py.

The flow is:

Nebula returns the model’s answer text.
The script pulls out the REASON: line with _extract_reason.
It computes simple counts with regex:
word counts for the answer, claim, and reason
entity-like tokens
digits
quotes
hedge words like “maybe” or “likely”
certainty words like “definitely”
It checks formatting:
whether LABEL: is present
whether REASON: is present
whether the response was truncated
It builds the final feature vector in _feature_row.
So the features are not learned embeddings or logits. They are transparent text signals computed from the model’s own output and the input claim.