"""
This script generates misclassified examples by replacing words with synonyms

This is the basic algorithm:
Get test examples where model correctly predicts agree or disagree.
For each example:
  For each word in body:
    Get class probabilities of example with word.
    Get class probabilities if word is replaced by a synonym.
    Save reduction in probability of correct class, and the synonym.
  Until the predicted label changes or we hit the max number of changes:
    Replace the word which causes the highest reduction in probability with its
        synonym.
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

    # Replace words with a synonym until the prediction changes (with a cap on the number of changes)
    changes = 0
    new_body = body
    synonyms = []
    while best_labels(model.predict_probabilities([headline], [new_body]))[0] == true_label_id:
        if (changes >= 10 or (not contributions)):
            return (body, []) # Could not change the label

        index, _contribution = contributions.pop() # Largest reduction
        token, pos = tagged_tokens[index]
        synonym = find_synonym(token, pos)

        if synonym:
            synonyms.append((index, token, synonym))
            tokens[index] = synonym
            new_body = DETOKENIZER.detokenize(tokens)
            changes += 1

    for _i, original, synonym in synonyms:
        print("Replacing {} with {}".format(original, synonym, index))

    # If the loop terminated then we were able to change the label
    return (new_body, synonyms)

def write_csvs(transformed_examples):
    t = round(time())
    with open('data/{}baseline_bodies.csv'.format(t), 'w',encoding='utf-8') as csvfile:
        fieldnames = ['Body ID', 'articleBody', 'Original body ID']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in transformed_examples:
            writer.writerow(example)

    with open('data/{}baseline_stances.csv'.format(t), 'w',encoding='utf-8') as csvfile:
        fieldnames = ['Body ID', 'Headline', 'Stance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in transformed_examples:
            writer.writerow(example)

    with open('data/{}baseline_changes.csv'.format(t), 'w',encoding='utf-8') as csvfile:
        fieldnames = ['Body ID', 'Stance', 'Headline', 'articleBody', 'originalBody', 'changes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in transformed_examples:
            writer.writerow(example)

def main():
    # Load dataset
    dataset = DataSet(name="filtered_test", path="data")

    # Load model
    model = BaselineModel(joblib.load('kfold_trained.joblib'))
    cached_model = CachedModel('data/cache/baseline.pkl', model)

    changes_counts = []
    transformed_examples = []

    # Transform each example
    for new_body_id, stance in tqdm(enumerate(dataset.stances[:50])):
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
    print("Prediction changed count: {}".format(len(with_changes)))
    print("Median changes: {}".format(np.median(with_changes)))
    print("Mean changes: {}".format(np.mean(with_changes)))

    cached_model.save() # Save the cache
    write_csvs(transformed_examples)

if __name__ == "__main__":
    main()
