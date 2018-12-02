"""
This script generates misclassified examples by replacing words with synonyms

This is the basic algorithm:
Get test examples where model correctly predicts agree or disagree.
For each example:
  For each word in body:
    Get class probabilities of example with word.
    Get class probabilities if word is replaced by a synonym.
    Save reduction in probability of correct class, and the synonym.
  Until the predicted label changes or we hit the max number of deletions:
    Replace the word which causes the highest reduction in probability with its
        synonym.
"""
import csv

import numpy as np
import sklearn as sk

from nltk import pos_tag, pos_tag_sents, word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from sklearn.externals import joblib
from tqdm import tqdm
from sacremoses import MosesTokenizer, MosesDetokenizer

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

def find_synonym(token, pos):
    synsets = wn.synsets(token, pos=pos)

    if synsets:
        # Use the first synset.
        words = synsets[0].lemma_names()

        # Find the first word that is not our token.
        for word in words:
            if word != token:
                return word

detokenizer = MosesDetokenizer()
def probabilities_with_synonym(model, headline, tagged_tokens, token_id, synonym):
    tokens = [ w for w, _pos in tagged_tokens ]
    tokens[token_id] = synonym
    new_body = detokenizer.detokenize(tokens)
    new_x = feature_vec(headline, new_body)
    return model.predict_proba(new_x)[0]

PROBABILITIES_CACHE = {} # body_id -> list of probabilities after replacing each token
def probabilities_with_synonyms(model, tagged_tokens, headline, body_id):
    probabilities = PROBABILITIES_CACHE.get(body_id)

    if not probabilities:
        probabilities = []
        for i, (token, pos) in enumerate(tagged_tokens):
            if len(token) > 3: # || and pos == wn.ADV:
                synonym = find_synonym(token.lower(), pos)

                if synonym:
                    prob = probabilities_with_synonym(model, headline, tagged_tokens, i, synonym)
                    probabilities.append((i, synonym, prob))

        PROBABILITIES_CACHE[body_id] = probabilities

    return probabilities

def calculate_reductions(model, original_probabilities, tagged_tokens, headline, body_id, true_label_id):
    original_probability = original_probabilities[0][true_label_id]
    probabilities = probabilities_with_synonyms(model, tagged_tokens, headline, body_id)

    return [
        (i, synonym, original_probability - p[true_label_id])
        for i, synonym, p in probabilities
    ]

def translate_tag(penntag):
    """ The tagger uses different tags than wordnet. """
    map = {
        'NN':wn.NOUN,
        'JJ':wn.ADJ,
        'VB':wn.VERB,
        'RB':wn.ADV
    }

    return map.get(penntag[:2], wn.NOUN) # Use first 2 chars, and fall back to noun.

def tokenize_and_tag(body):
    sentence_tokens = [word_tokenize(sentence) for sentence in sent_tokenize(body)]
    tagged_sentences = pos_tag_sents(sentence_tokens)
    flattened = [tagged for sentence in tagged_sentences for tagged in sentence]
    return [(token, translate_tag(tag)) for token, tag in flattened]

def construct_example(model, original_x, body, body_id, headline, true_label_id):
    original_probabilities = model.predict_proba(original_x.reshape(1, -1))
    tagged_tokens = tokenize_and_tag(body)

    # For each word calculate the class probability reduction if it is removed
    reductions = calculate_reductions(model, original_probabilities, tagged_tokens, headline, body_id, true_label_id)
    reductions.sort(key=lambda x: x[2]) # Lowest first

    # Replace words until the prediction changes (with a cap on the number of changes)
    changes = 0
    new_body = body
    new_tokens = [w for w, _pos in tagged_tokens]
    replacements = []
    while model.predict(feature_vec(headline, new_body)) == true_label_id:
        if (changes >= 20 or (not reductions)):
            return (body, 0) # Could not change the label

        index, synonym, _reduction = reductions.pop() # Largest reduction
        replacements.append((index, synonym))
        new_tokens[index] = synonym
        new_body = detokenizer.detokenize(new_tokens)
        changes += 1

    for index, synonym in replacements:
        print("Replacing {} with {} at index {}".format(tagged_tokens[index][0], synonym, index))

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
    for index in tqdm(correctly_predicted[:50]):
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
    print("Median changes: {}".format(np.median(with_changes)))

    write_csvs(transformed_examples)

if __name__ == "__main__":
    main()
