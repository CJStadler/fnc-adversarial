import heapq
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

def calculate_reductions(model, original_probabilities, body_tokens, headline, true_label_id):
    reductions = []

    for i, token in enumerate(body_tokens):
        tokens_without = body_tokens.copy()
        del tokens_without[i]
        body_without = " ".join(tokens_without)
        x_without = feature_vec(headline, body_without)
        probabilities = model.predict_proba(x_without)
        reduction = original_probabilities[0][true_label_id] - probabilities[0][true_label_id]

        reductions.append((i, reduction))

    return reductions

def construct_example(model, original_x, body, headline, true_label_id):
    original_probabilities = model.predict_proba(original_x.reshape(1, -1))
    body_tokens = body.split(" ")

    # For each word calculate the class probability reduction if it is removed
    reductions = calculate_reductions(model, original_probabilities, body_tokens, headline, true_label_id)
    reductions.sort(key=lambda x: -x[1]) # Lowest first

    # Remove words until the prediction changes (with a cap on the number of changes)
    changes = 0
    new_body = body
    removed_so_far = [] # indices
    while model.predict(feature_vec(headline, new_body)) == true_label_id:
        if (changes >= 10):
            return (body, 0) # Could not change the label

        index, _reduction = reductions.pop() # Largest reduction
        removed_so_far.append(index)
        new_tokens = [t for i, t in enumerate(body_tokens) if i not in removed_so_far]
        new_body = " ".join(new_tokens)
        changes += 1

    # If the loop terminated then we were able to change the label
    return (new_body, changes)

def write_csv(transformed_examples):
    with open('transformed_examples.csv', 'w') as csvfile:
        fieldnames = ['Body ID', 'articleBody', 'deletions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

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
    model = joblib.load('kfold_trained.joblib')

    # Make predictions
    predicted = [LABELS[int(a)] for a in model.predict(X_test)]
    actual = [LABELS[int(a)] for a in y]

    # Select correct predictions of agree or disagree
    correct_agree_disagree = []

    for i, (prediction, truth) in enumerate(zip(predicted, actual)):
        if ((prediction == "disagree" or prediction == "agree") and prediction == truth):
            correct_agree_disagree.append(i)

    change_counts = []

    # Transform each example
    for index in tqdm(correct_agree_disagree):
        headline = headlines[index]
        body = bodies[index]
        body_id = body_ids[index]
        true_label_id = y[index]
        original_x = X_test[index]

        new_body, deletions = construct_example(model, original_x, body, headline, true_label_id)
        transformed_examples.append({ "Body ID": body_id, "articleBody": new_body, "deletions": deletions})
        change_counts.append(deletions)

    with_changes = [c for c in change_counts if c > 0]
    print("Original correct agree/disagree: {}".format(len(correct_agree_disagree)))
    print("Prediction changed count: {}".format(len(with_changes)))
    print("Median deletions: {}".format(np.median(with_changes)))

    write_csv(transformed_examples)


if __name__ == "__main__":
    main()
