# Crafting Adversarial Input Sequences for Recurrent Neural Networks

- For every word of the sequence calculate "the direction in which
  we have to perturb each of the word embedding components
  in order to reduce the probability assigned to the current class,
  and thus change the class assigned to the sentence."
- Then choose a word z from the dictionary "such that the sign of the difference
  between the embeddings of z and the original input word is closest" to the
  desired direction.
- White-box.
