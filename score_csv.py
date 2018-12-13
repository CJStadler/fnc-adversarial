import sys
import numpy as np

from csv import DictReader
import argparse

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
    parser = argparse.ArgumentParser(description='Score model on transformed set')
    parser.add_argument("-b","--baseline", required=True, help="timestamp of baselinen stances file")
    parser.add_argument("-t","--target", required=True, help="uclmr or athene")
    args = parser.parse_args()
    actual = get_truth("{}_baseline".format(args.baseline))

    print("Scores on the original set")
    predicted = read_predictions("{}_filtered_predictions".format(args.target))
    report_score(actual, predicted)

    print("Scores on the transformed set")
    predicted = read_predictions("{}_synonym_predictions".format(args.target))
    report_score(actual, predicted)
