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
        self._init_cache(filename)

    def save():
        with open(filename, 'wb') as file:
            pickle.dump(self.cache, file)

    def predict_probabilities(self, headlines, bodies):
        # TODO: oh no we need to get any that are cached
        probs = self.model.predict_probabilities(headlines, bodies)
        self._cache_probabilities(probs, headlines, bodies)
        return probs

    def _cache_probabilities(self, probabilities, headlines, bodies):
        for probs, headline, body in zip(probabilities, headlines, bodies):


    def _single_predict(self, headline, body):
        h = self._hash(headline, body)
        predictions = self.cache.get(h)

        if not predictions:
            predictions = self.model.predict_probabilities([headline], [body])
            self.cache[h] = predictions

        return predictions

    def _hash(self, headline, body):
        h = md5()
        h.update(headline.encode('utf-8'))
        h.update(body.encode('utf-8'))
        return h.digest()

    def _init_cache(self, filename):
        path = Path(filename)

        if path.is_file():
            with path.open(mode='rb') as file:
                self.cache = pickle.load(file)
        else:
            self.cache = {}
