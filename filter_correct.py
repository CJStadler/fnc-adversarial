"""
This script selects examples which are correctly classified.
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

def write_csvs(examples):
    with open('data/filtered_test_bodies.csv', 'w') as csvfile:
        fieldnames = ['Body ID', 'articleBody']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in examples:
            writer.writerow(example)

    with open('data/filtered_test_stances.csv', 'w') as csvfile:
        fieldnames = ['Body ID', 'Headline', 'Stance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for example in examples:
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

    examples = []

    for index in tqdm(correctly_predicted):
        headline = headlines[index]
        body = bodies[index]
        body_id = body_ids[index]
        true_label_id = y[index]

        examples.append({
            "Body ID": index,
            "articleBody": body,
            "Stance": LABELS[true_label_id],
            "Headline": headline
        })

    write_csvs(examples)

if __name__ == "__main__":
    main()
