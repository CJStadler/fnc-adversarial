"""
This script generates misclassified examples by replacing words with synonyms

This is the basic algorithm:
For each example:
  For each word in body:
    Get class probabilities of example with word.
    Get class probabilities if word is replaced by a synonym.
    Save reduction in probability of correct class, and the synonym.
  Replace the N_CHANGES words which cause the highest reductions in probability
    with their synonyms.
"""
import csv
from joblib import Parallel, delayed
import argparse

from time import time
from sklearn.externals import joblib
from tqdm import tqdm
from sacremoses import MosesTokenizer, MosesDetokenizer

from utils.nlp import find_synonym, tokenize_and_tag
from models import BaselineModel, CachedModel
from nltk.corpus import wordnet as wn
from utils.synonyms import construct_example

DETOKENIZER = MosesDetokenizer()
N_CHANGES = 4

# Load model
model = BaselineModel(joblib.load('kfold_trained.joblib'))
cached_model = CachedModel('data/cache/baseline.pkl', model)
shared_list = list()

# pre-load WordNet
print(find_synonym("adversarial",wn.ADJ))

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

def main(num_jobs, begin, end, v):
    print('Replacing {} words in each example'.format(N_CHANGES))
    changes_counts = []

    # Transform each example
    transformed_examples = Parallel(n_jobs=num_jobs, verbose=v)(delayed(construct_example)(i) for i in range(begin,end,1))

    with_changes = [c for c in changes_counts if c > 0]

    cached_model.save() # Save the cache
    write_csvs(transformed_examples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Adversarial samples for the FNC-1 dataset')
    parser.add_argument("-n","--num_jobs",type=int, default=2, help="number of parallel jobs.")
    parser.add_argument("-i","--begin",type=int, default=0, help="start with item number i (>0)")
    parser.add_argument("-j","--end",type=int, default=50, help="stop with item number j (<7000)")
    parser.add_argument("-v","--verbose",type=int, default=1, help="joblib verbosity, 1-10 is the useful range")
    args = parser.parse_args()
    main(args.num_jobs, args.begin, args.end, args.verbose)
