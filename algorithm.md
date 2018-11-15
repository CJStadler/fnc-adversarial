- Get test examples where model correctly predicts agree or disagree.
- For each example.
  - For each word.
    - Get class probabilities of example with and without word.
    - Save change in probability of correct class.
  - Select N words with greatest reduction in correct class probability and
    remove them from example.
