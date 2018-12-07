import pickle

from hashlib import md5
from pathlib import Path

class CachedModel:
    """
    This wraps another model and caches its predictions. The cache key is a
    hash of the headline and body.

    The cache is loaded from the given file, and save() should be called when
    finished.
    """
    def __init__(self, filename, model):
        self.model = model
        self.filename = filename
        self.cache = self._load_cache(filename)

    def save(self):
        with open(self.filename, 'wb') as file:
            pickle.dump(self.cache, file)

    def predict_probabilities(self, headlines, bodies):
        """
        Query the model only for headline/body pairs that are not in the cahce.
        """
        hashes = [self._hash(h, b) for h, b in zip(headlines, bodies)]
        all_predictions = [self.cache.get(h) for h in hashes] # missing will have None

        missing_ids = [i for i, p in enumerate(all_predictions) if p is None]

        if len(missing_ids) > 0:
            missing_headlines = [headlines[i] for i in missing_ids]
            missing_bodies = [bodies[i] for i in missing_ids]

            new_predictions = self.model.predict_probabilities(missing_headlines, missing_bodies)

            # Add new predictions to the cache and the array of all predictions
            for i, p in zip(missing_ids, new_predictions):
                self.cache[hashes[i]] = p
                all_predictions[i] = p

        return all_predictions

    def _hash(self, headline, body):
        h = md5()
        h.update(headline.encode('utf-8'))
        h.update(body.encode('utf-8'))
        return h.digest()

    def _load_cache(self, filename):
        path = Path(filename)

        if path.is_file():
            with path.open(mode='rb') as file:
                return pickle.load(file)
        else:
            return {}
