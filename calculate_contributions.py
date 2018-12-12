from sklearn.externals import joblib
from fnc_1_baseline.utils.dataset import DataSet
from fnc_1_baseline.utils.score import LABELS

from utils.nlp import tokenize
from utils.contributions import calculate_contributions
from models import BaselineModel, CachedModel

def main():
    # Load model
    model = BaselineModel(joblib.load('kfold_trained.joblib'))
    cached_model = CachedModel('data/cache/baseline.pkl', model)

    body_id = 2380
    headline = "Mystery of 50ft giant crab caught on camera in Kent harbour"
    stance = "discuss"

    print('Stance: {}'.format(stance))
    print('Headline: "{}"'.format(headline))

    test_dataset = DataSet(name="competition_test", path="fnc_1_baseline/fnc-1")

    body = test_dataset.articles[body_id]
    true_label_id = LABELS.index(stance)

    tokens = tokenize(body)

    original_probabilities = model.predict_probabilities([headline], [body])[0]

    contributions = calculate_contributions(cached_model, original_probabilities, tokens, headline, body_id, true_label_id)
    by_index = {}

    for i, c in contributions:
        by_index[i] = c

    with_contributions = [(t, by_index.get(i)) for i, t in enumerate(tokens)]
    print(with_contributions)

if __name__ == "__main__":
    main()
