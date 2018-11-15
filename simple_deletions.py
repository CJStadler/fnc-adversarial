import numpy as np
import sklearn as sk
from sklearn.externals import joblib

from fnc_1_baseline.utils.dataset import DataSet
from fnc_1_baseline.feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from fnc_1_baseline.feature_engineering import word_overlap_features
from fnc_1_baseline.feature_engineering import clean
from fnc_1_baseline.utils.score import report_score, LABELS, score_submission

def tokens(text):
    input.split(' ')

model = joblib.load('kfold_trained.joblib')

# Load test data
test_dataset = DataSet(name="competition_test", path="fnc_1_baseline/fnc-1")
headlines, bodies, body_ids, y = [],[],[],[]

for stance in test_dataset.stances:
    y.append(LABELS.index(stance['Stance']))
    headlines.append(stance['Headline'])
    body_ids.append(stance['Body ID'])
    bodies.append(test_dataset.articles[stance['Body ID']])

X_overlap = gen_or_load_feats(word_overlap_features, headlines, bodies, "fnc_1_baseline/features/overlap.competition.npy")
X_refuting = gen_or_load_feats(refuting_features, headlines, bodies, "fnc_1_baseline/features/refuting.competition.npy")
X_polarity = gen_or_load_feats(polarity_features, headlines, bodies, "fnc_1_baseline/features/polarity.competition.npy")
X_hand = gen_or_load_feats(hand_features, headlines, bodies, "fnc_1_baseline/features/hand.competition.npy")

X_test = np.c_[X_hand, X_polarity, X_refuting, X_overlap]

# Make predictions
predicted = [LABELS[int(a)] for a in best_fold.predict(X_test)]
actual = [LABELS[int(a)] for a in y]

print(len(predicted))
print(len(actual))
