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

from sklearn.externals import joblib
from sacremoses import MosesTokenizer, MosesDetokenizer
from fnc_1_baseline.utils.dataset import DataSet
from fnc_1_baseline.utils.score import LABELS

from utils.nlp import find_synonym, tokenize_and_tag
from utils.contributions import calculate_contributions
from models import BaselineModel, CachedModel

DETOKENIZER = MosesDetokenizer()

# Load dataset
dataset = DataSet(name="filtered_test", path="data")

# Load model
model = BaselineModel(joblib.load('kfold_trained.joblib'))
cached_model = CachedModel('data/cache/baseline.pkl', model)

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

def construct_example(stance_id, N_CHANGES):
    stance  = dataset.stances[stance_id]
    new_body_id = stance_id
    headline = stance['Headline']
    body_id = stance['Body ID']
    original_body = dataset.articles[body_id]
    true_label_id = LABELS.index(stance['Stance'])
    original_probabilities = cached_model.predict_probabilities([headline], [original_body])[0]
    tagged_tokens = tokenize_and_tag(original_body)
    tokens = [w for w, _pos in tagged_tokens]

    # For each word calculate its contribution to the class probability.
    contributions = calculate_contributions(cached_model, original_probabilities, tokens, headline, body_id, true_label_id)
    contributions.sort(key=lambda x: x[1]) # Lowest first

    changes = 0
    new_body = original_body
    synonyms = []

    while changes < N_CHANGES:
        if not contributions:
            break

        index, _contribution = contributions.pop() # Largest reduction
        token, pos = tagged_tokens[index]
        synonym = find_synonym(token, pos)

        if synonym:
            # print("Replacing {} with {}".format(token, synonym, index))
            synonyms.append((index, token, synonym))
            tokens[index] = synonym
            changes += 1

    new_body = DETOKENIZER.detokenize(tokens)

    # If the loop terminated then we were able to change the label
    stance['changes'] = synonyms
    stance["Body ID"] = new_body_id
    stance["articleBody"] = new_body
    stance["Original body ID"] = body_id
    stance["originalBody"] = original_body
    return stance
