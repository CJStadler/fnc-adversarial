import sys
import numpy as np

from csv import DictReader

from fnc_1_baseline.utils.dataset import DataSet
from fnc_1_baseline.utils.score import report_score, LABELS

def get_truth(name):
    dataset = DataSet(name=name, path="data")
    return [stance['Stance'] for stance in dataset.stances]

def read_predictions(name):
    fn = "data/{}.csv".format(name)

    rows = []
    with open(fn, "r", encoding='utf-8') as file:
        r = DictReader(file)

        for line in r:
            rows.append(line)

    return [row['Stance'] for row in rows]

if __name__ == "__main__":
    actual = get_truth("synonym_baseline")

    print("Scores on the original set")
    predicted = read_predictions("athene_predictions_original")
    report_score(actual, predicted)

    print("Scores on the transformed set")
    predicted = read_predictions("athene_predictions_transformed")
    report_score(actual, predicted)
