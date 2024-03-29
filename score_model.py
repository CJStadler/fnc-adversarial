import sys
import numpy as np
import argparse

from sklearn.externals import joblib

from fnc_1_baseline.feature_engineering import refuting_features, polarity_features, hand_features
from fnc_1_baseline.feature_engineering import word_overlap_features, gen_or_load_feats
from fnc_1_baseline.utils.dataset import DataSet
from fnc_1_baseline.utils.score import report_score, LABELS


def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "fnc_1_baseline/features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "fnc_1_baseline/features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "fnc_1_baseline/features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "fnc_1_baseline/features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

def score_dataset(model, name):
    dataset = DataSet(name=name, path="data")
    X, y = generate_features(dataset.stances, dataset, name)

    predicted = [LABELS[int(a)] for a in model.predict(X)]
    actual = [LABELS[int(a)] for a in y]

    report_score(actual,predicted)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score model on transformed set')
    parser.add_argument("-b","--baseline", required=True, help="prefix of baseline stances file")
    args = parser.parse_args()
    model = joblib.load('kfold_trained.joblib')

    print("\n\tBaseline model performance:")
    print("Scores on the original set")
    score_dataset(model, "filtered_test")
    print("\nScores on the transformed set")
    score_dataset(model, "{}_baseline".format(args.baseline))
