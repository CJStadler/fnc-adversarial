"""
This script generates misclassified examples by replacing words with synonyms

This is the basic algorithm:
For each example:
  For each word in body:
    Get class probabilities of example with word.
    Get class probabilities if word is replaced by a synonym.
    Save reduction in probability of correct class, and the synonym.
  Replace the N_CHANGES words which cause the highest reductions in probability
    with their synonyms.
"""
import csv

import numpy as np
import sklearn as sk

from time import time
from sklearn.externals import joblib
from tqdm import tqdm
from sacremoses import MosesTokenizer, MosesDetokenizer
from fnc_1_baseline.utils.dataset import DataSet
from fnc_1_baseline.utils.score import report_score, LABELS, score_submission

from utils.nlp import find_synonym, tokenize_and_tag
from utils.contributions import calculate_contributions
from models import BaselineModel, CachedModel

DETOKENIZER = MosesDetokenizer()
N_CHANGES = 4

def get_label(probabilities):
    """
    Takes an array of probabilities and returns the index of the highest.
    """
    return max(enumerate(probabilities), key=lambda x: x[1])[0]

def best_labels(probabilities):
    """
    Takes an array of arrays of probabilities and returns an array of the index
    of the max of each sub-array.
    """
    return [get_label(probs) for probs in probabilities]

def construct_example(model, body, body_id, headline, true_label_id):
    original_probabilities = model.predict_probabilities([headline], [body])[0]
    tagged_tokens = tokenize_and_tag(body)
    tokens = [w for w, _pos in tagged_tokens]

    # For each word calculate its contribution to the class probability.
    contributions = calculate_contributions(model, original_probabilities, tokens, headline, body_id, true_label_id)
    contributions.sort(key=lambda x: x[1]) # Lowest first

    changes = 0
    new_body = body
    synonyms = []

    while changes < N_CHANGES:
        if not contributions:
            break

        index, _contribution = contributions.pop() # Largest reduction
        token, pos = tagged_tokens[index]
        synonym = find_synonym(token, pos)

        if synonym:
            # print("Replacing {} with {}".format(token, synonym, index))
            synonyms.append((index, token, synonym))
            tokens[index] = synonym
            changes += 1

    new_body = DETOKENIZER.detokenize(tokens)

    # If the loop terminated then we were able to change the label
    return (new_body, synonyms)

def write_csvs(transformed_examples):
    t = round(time())
    with open('data/{}baseline_bodies.csv'.format(t), 'w') as csvfile:
        fieldnames = ['Body ID', 'articleBody', 'Original body ID']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in transformed_examples:
            writer.writerow(example)

    with open('data/{}baseline_stances.csv'.format(t), 'w') as csvfile:
        fieldnames = ['Body ID', 'Headline', 'Stance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in transformed_examples:
            writer.writerow(example)

    with open('data/{}baseline_changes.csv'.format(t), 'w') as csvfile:
        fieldnames = ['Body ID', 'Stance', 'Headline', 'articleBody', 'originalBody', 'changes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in transformed_examples:
            writer.writerow(example)

def main():
    # Load dataset
    dataset = DataSet(name="filtered_test", path="data")

    print('Replacing {} words in each example'.format(N_CHANGES))

    # Load model
    model = BaselineModel(joblib.load('kfold_trained.joblib'))
    cached_model = CachedModel('data/cache/baseline.pkl', model)

    changes_counts = []
    transformed_examples = []

    # Transform each example
    stances = [(i, stance) for i, stance in enumerate(dataset.stances)]
    for new_body_id, stance in tqdm(stances):
        try:
            headline = stance['Headline']
            body_id = stance['Body ID']
            original_body = dataset.articles[body_id]
            true_label_id = LABELS.index(stance['Stance'])

            new_body, changes = construct_example(cached_model, original_body, body_id, headline, true_label_id)
            transformed_examples.append({
                "Body ID": new_body_id,
                "articleBody": new_body,
                "Stance": LABELS[true_label_id],
                "Headline": headline,
                "Original body ID": body_id,
                "originalBody": original_body,
                "changes": changes,
            })
            changes_counts.append(len(changes))
        except Exception as e:
            print("Error for row {}: {}".format(new_body_id, e))

    with_changes = [c for c in changes_counts if c > 0]

    cached_model.save() # Save the cache
    write_csvs(transformed_examples)

if __name__ == "__main__":
    main()
