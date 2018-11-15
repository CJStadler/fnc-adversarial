import heapq

import numpy as np
import sklearn as sk
from sklearn.externals import joblib

from fnc_1_baseline.utils.dataset import DataSet
from fnc_1_baseline.feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from fnc_1_baseline.feature_engineering import word_overlap_features
from fnc_1_baseline.feature_engineering import clean
from fnc_1_baseline.utils.score import report_score, LABELS, score_submission

def generate_features(headlines, bodies):
    X_overlap = gen_or_load_feats(word_overlap_features, headlines, bodies, "fnc_1_baseline/features/overlap.competition.npy")
    X_refuting = gen_or_load_feats(refuting_features, headlines, bodies, "fnc_1_baseline/features/refuting.competition.npy")
    X_polarity = gen_or_load_feats(polarity_features, headlines, bodies, "fnc_1_baseline/features/polarity.competition.npy")
    X_hand = gen_or_load_feats(hand_features, headlines, bodies, "fnc_1_baseline/features/hand.competition.npy")

    return np.c_[X_hand, X_polarity, X_refuting, X_overlap]

def feature_vec(headline, body):
    """
    Construct a feature vector using a single example of headline and body.
    """
    return generate_features([headline], [body])

# Load test data
test_dataset = DataSet(name="competition_test", path="fnc_1_baseline/fnc-1")
headlines, bodies, body_ids, y = [],[],[],[]

for stance in test_dataset.stances:
    y.append(LABELS.index(stance['Stance']))
    headlines.append(stance['Headline'])
    body_ids.append(stance['Body ID'])
    bodies.append(test_dataset.articles[stance['Body ID']])

X_test = generate_features(headlines, bodies)

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

# Transform each example
for index in correct_agree_disagree[:2]:
    headline = headlines[index]
    body = bodies[index]
    body_id = body_ids[index]
    true_label_id = y[index]

    original_x = X_test[index]
    original_probabilities = model.predict_proba(original_x.reshape(1, -1))

    # For each word calculate the class probability reduction if it is removed
    body_tokens = body.split(" ")
    reductions = []

    for i, token in enumerate(body_tokens):
        tokens_without = body_tokens.copy()
        del tokens_without[i]
        body_without = " ".join(tokens_without)
        x_without = feature_vec(headline, body)
        probabilities = model.predict_proba(x_without)
        reduction = original_probabilities[true_label_id] - probabilities[true_label_id]
        reductions.append((i, reduction))

    greatest_reductions = heapq.nlargest(10, reductions, key=lambda x: x[1])
    reduction_ids = [i for i, r in greatest_reductions]

    # Build a new body without the tokens that produced the greatest reductions.
    new_tokens = [t for i, t in enumerate(body_tokens) if i not in reduction_ids]
    new_body = " ".join(new_tokens)

    import pdb; pdb.set_trace()
    print(new_body)
