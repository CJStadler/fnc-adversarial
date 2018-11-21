"""
This script generates misclassified examples by deleting words from examples
that were correctly classified by the baseline model.

This is the basic algorithm:
Get test examples where model correctly predicts agree or disagree.
For each example:
  For each word in body:
    Get class probabilities of example with and without word.
    Save reduction in probability of correct class.
  Until the predicted label changes or we hit the max number of deletions:
    Delete the word which causes the highest reduction in probability.
"""
import csv

import numpy as np
import sklearn as sk

from sklearn.externals import joblib
from tqdm import tqdm

from fnc_1_baseline.utils.dataset import DataSet
from fnc_1_baseline.feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from fnc_1_baseline.feature_engineering import word_overlap_features
from fnc_1_baseline.feature_engineering import clean
from fnc_1_baseline.utils.score import report_score, LABELS, score_submission

def load_trained_model():
    return joblib.load('kfold_trained.joblib')

def generate_test_features(headlines, bodies):
    overlap = gen_or_load_feats(word_overlap_features, headlines, bodies, "fnc_1_baseline/features/overlap.competition.npy")
    refuting = gen_or_load_feats(refuting_features, headlines, bodies, "fnc_1_baseline/features/refuting.competition.npy")
    polarity = gen_or_load_feats(polarity_features, headlines, bodies, "fnc_1_baseline/features/polarity.competition.npy")
    hand = gen_or_load_feats(hand_features, headlines, bodies, "fnc_1_baseline/features/hand.competition.npy")

    return np.c_[hand, polarity, refuting, overlap]

def feature_vec(headline, body):
    """
    Construct a feature vector using a single example of headline and body.
    """
    headline = [headline]
    body = [body]
    overlap = word_overlap_features(headline, body)
    refuting = refuting_features(headline, body)
    polarity = polarity_features(headline, body)
    hand = hand_features(headline, body)

    return np.c_[hand, polarity, refuting, overlap]


def probabilities_without_token(model, headline, body_tokens, token_id):
    tokens_without = body_tokens.copy()
    del tokens_without[token_id]
    body_without = " ".join(tokens_without)
    x_without = feature_vec(headline, body_without)
    return model.predict_proba(x_without)[0]

PROBABILITIES_CACHE = {} # body_id -> list of probabilities after deleting each token
def probabilities_without_tokens(model, body_tokens, headline, body_id):
    probabilities = PROBABILITIES_CACHE.get(body_id)

    if not probabilities:
        probabilities = []
        for i, token in enumerate(body_tokens):
            if len(token) > 3:
                prob = probabilities_without_token(model, headline, body_tokens, i)
                probabilities.append((i, prob))

        PROBABILITIES_CACHE[body_id] = probabilities

    return probabilities

def calculate_reductions(model, original_probabilities, body_tokens, headline, body_id, true_label_id):
    original_probability = original_probabilities[0][true_label_id]
    probabilities = probabilities_without_tokens(model, body_tokens, headline, body_id)

    return [
        (i, original_probability - p[true_label_id])
        for i, p in probabilities
    ]

def construct_example(model, original_x, body, body_id, headline, true_label_id):
    original_probabilities = model.predict_proba(original_x.reshape(1, -1))
    body_tokens = body.split(" ")

    # For each word calculate the class probability reduction if it is removed
    reductions = calculate_reductions(model, original_probabilities, body_tokens, headline, body_id, true_label_id)
    reductions.sort(key=lambda x: x[1]) # Lowest first

    # Remove words until the prediction changes (with a cap on the number of changes)
    changes = 0
    new_body = body
    removed_so_far = [] # indices
    while model.predict(feature_vec(headline, new_body)) == true_label_id:
        if (changes >= 10 or (not reductions)):
            return (body, 0) # Could not change the label

        index, _reduction = reductions.pop() # Largest reduction
        removed_so_far.append(index)
        new_tokens = [t for i, t in enumerate(body_tokens) if i not in removed_so_far]
        new_body = " ".join(new_tokens)
        changes += 1

    # If the loop terminated then we were able to change the label
    return (new_body, changes)

def write_csvs(transformed_examples):
    with open('data/transformed_test_bodies.csv', 'w') as csvfile:
        fieldnames = ['Body ID', 'articleBody', 'deletions', 'Original body ID']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in transformed_examples:
            writer.writerow(example)

    with open('data/transformed_test_stances.csv', 'w') as csvfile:
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

    X_test = generate_test_features(headlines, bodies)

    # Load model
    model = load_trained_model()

    # Make predictions
    predictions =  model.predict(X_test)

    # Select correct predictions of agree or disagree
    correctly_predicted = []

    for i, (prediction, truth) in enumerate(zip(predictions, y)):
        if (prediction == truth and (prediction in [0, 1, 2])): # agree, disagree, or discuss
            correctly_predicted.append(i)

    change_counts = []

    # correctly_predicted = correctly_predicted[:50]
    print("Original correct: {}".format(len(correctly_predicted)))

    # Transform each example
    for index in tqdm(correctly_predicted):
        try:
            headline = headlines[index]
            original_body = bodies[index]
            body_id = body_ids[index]
            true_label_id = y[index]
            original_x = X_test[index]

            new_body, deletions = construct_example(model, original_x, original_body, body_id, headline, true_label_id)
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
    print("Median deletions: {}".format(np.median(with_changes)))

    write_csvs(transformed_examples)

if __name__ == "__main__":
    main()
