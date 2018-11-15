# Towards Crafting Text Adversarial Samples

- "The main contribution of this work is to craft adversarial samples in the
  domain of text data by preserving the semantic meaning of the sentences as
  much as possible"
- Use pool of synonyms and misspellings (but which are also valid words, to
  avoid spell-checkers).
- Remove adverbs because they don't alter structure.
- Replace with same part of speech.
- Minimum number of changes to switch classification.
