import numpy as np
from fnc_1_baseline.feature_engineering import refuting_features, polarity_features, hand_features, word_overlap_features, gen_or_load_feats

class BaselineModel:
    def __init__(self, model):
        self.model = model

    def predict_probabilities(self, headlines, bodies):
        """
        Takes an array of headlines and an array of bodies, both of length N.
        Returns an array of length N where the ith element is an array of the
        class probabilities for the ith headline and ith body.
        """
        X = self._generate_features(headlines, bodies)
        return self.model.predict_proba(X)

    def _generate_features(self, headlines, bodies):
        overlap = gen_or_load_feats(word_overlap_features, headlines, bodies, "fnc_1_baseline/features/overlap.competition.npy")
        refuting = gen_or_load_feats(refuting_features, headlines, bodies, "fnc_1_baseline/features/refuting.competition.npy")
        polarity = gen_or_load_feats(polarity_features, headlines, bodies, "fnc_1_baseline/features/polarity.competition.npy")
        hand = gen_or_load_feats(hand_features, headlines, bodies, "fnc_1_baseline/features/hand.competition.npy")

        return np.c_[hand, polarity, refuting, overlap]
