"""
This script generates misclassified examples by replacing words with synonyms

This is the basic algorithm:
Get test examples where model correctly predicts agree or disagree.
For each example:
  For each word in body:
    Get class probabilities of example with word.
    Get class probabilities if word is replaced by a synonym.
    Save reduction in probability of correct class, and the synonym.
  Until the predicted label changes or we hit the max number of deletions:
    Replace the word which causes the highest reduction in probability with its
        synonym.
"""
import csv

import numpy as np
import sklearn as sk

from sklearn.externals import joblib
from tqdm import tqdm
from sacremoses import MosesTokenizer, MosesDetokenizer

from fnc_1_baseline.utils.dataset import DataSet
from fnc_1_baseline.utils.score import report_score, LABELS, score_submission

from nlp import find_synonym, tokenize_and_tag
from models import BaselineModel, CachedModel

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

detokenizer = MosesDetokenizer()
def probabilities_with_synonym(model, headline, tagged_tokens, token_id, synonym):
    tokens = [ w for w, _pos in tagged_tokens ]
    tokens[token_id] = synonym
    new_body = detokenizer.detokenize(tokens)
    return model.predict_proba([headline], [new_body])[0]

SYNONYMS_CACHE = {} # body_id -> list of synonym for each token
def get_synonyms(tagged_tokens, body_id):
    synonyms = SYNONYMS_CACHE.get(body_id)

    if not synonyms:
        synonyms = [
            find_synonym(token, pos) if len(token) > 3 else None
            for token, pos in tagged_tokens
        ]
        SYNONYMS_CACHE[body_id] = synonyms

    return synonyms

def probabilities_with_synonyms(model, tagged_tokens, headline, body_id):
    synonyms = get_synonyms(tagged_tokens, body_id)

    probabilities = []

    for i, synonym in enumerate(synonyms):
        if synonym:
            prob = probabilities_with_synonym(model, headline, tagged_tokens, i, synonym)
            probabilities.append((i, synonym, prob))

    return probabilities

def calculate_reductions(model, original_probabilities, tagged_tokens, headline, body_id, true_label_id):
    original_probability = original_probabilities[0][true_label_id]
    probabilities = probabilities_with_synonyms(model, tagged_tokens, headline, body_id)

    return [
        (i, synonym, original_probability - p[true_label_id])
        for i, synonym, p in probabilities
    ]

def construct_example(model, body, body_id, headline, true_label_id):
    original_probabilities = model.predict_proba([headline], [body])[0]
    tagged_tokens = tokenize_and_tag(body)

    # For each word calculate the class probability reduction if it is removed
    reductions = calculate_reductions(model, original_probabilities, tagged_tokens, headline, body_id, true_label_id)
    reductions.sort(key=lambda x: x[2]) # Lowest first

    # Replace words until the prediction changes (with a cap on the number of changes)
    changes = 0
    new_body = body
    new_tokens = [w for w, _pos in tagged_tokens]
    replacements = []
    while best_labels(model.predict_probabilities([headline], [new_body]))[0] == true_label_id:
        if (changes >= 20 or (not reductions)):
            return (body, 0) # Could not change the label

        index, synonym, _reduction = reductions.pop() # Largest reduction
        replacements.append((index, synonym))
        new_tokens[index] = synonym
        new_body = detokenizer.detokenize(new_tokens)
        changes += 1

    for index, synonym in replacements:
        print("Replacing {} with {} at index {}".format(tagged_tokens[index][0], synonym, index))

    # If the loop terminated then we were able to change the label
    return (new_body, changes)

def write_csvs(transformed_examples):
    t = round(time())
    with open('data/{}transformed_test_bodies.csv'.format(t), 'w') as csvfile:
        fieldnames = ['Body ID', 'articleBody', 'deletions', 'Original body ID']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in transformed_examples:
            writer.writerow(example)

    with open('data/{}transformed_test_stances.csv'.format(t), 'w') as csvfile:
        fieldnames = ['Body ID', 'Headline', 'Stance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in transformed_examples:
            writer.writerow(example)

def main():
    # Load test data
    test_dataset = DataSet(name="competition_test", path="fnc_1_baseline/fnc-1")
    headlines, bodies, body_ids, y = [],[],[],[]
    transformed_examples = []

    for stance in test_dataset.stances:
        y.append(LABELS.index(stance['Stance']))
        headlines.append(stance['Headline'])
        body_ids.append(stance['Body ID'])
        bodies.append(test_dataset.articles[stance['Body ID']])

    # Load model
    model = BaselineModel(joblib.load('kfold_trained.joblib'))
    model = CachedModel('data/cache/baseline.pkl', model)
    import pdb; pdb.set_trace()
    # Make predictions
    predictions = best_labels(model.predict_probabilities(headlines, bodies))

    import pdb; pdb.set_trace()
    model.save()
    # Select correct predictions of agree or disagree
    correctly_predicted = []

    for i, (prediction, truth) in enumerate(zip(predictions, y)):
        if (prediction == truth and (prediction in [0, 1, 2])): # agree, disagree, or discuss
            correctly_predicted.append(i)

    change_counts = []

    correctly_predicted = correctly_predicted[:3]
    print("Original correct: {}".format(len(correctly_predicted)))

    # Transform each example
    for index in tqdm(correctly_predicted):
        try:
            headline = headlines[index]
            original_body = bodies[index]
            body_id = body_ids[index]
            true_label_id = y[index]

            new_body, deletions = construct_example(model, original_body, body_id, headline, true_label_id)
            transformed_examples.append({
                "Body ID": index,
                "articleBody": new_body,
                "Stance": LABELS[true_label_id],
                "Headline": headline,
                "Original body ID": body_id,
                "deletions": deletions,
            })
            change_counts.append(deletions)
        except Exception as e:
            print("Error for {}: {}".format(index, e))

    with_changes = [c for c in change_counts if c > 0]
    print("Prediction changed count: {}".format(len(with_changes)))
    print("Median changes: {}".format(np.median(with_changes)))

    write_csvs(transformed_examples)

if __name__ == "__main__":
    main()
