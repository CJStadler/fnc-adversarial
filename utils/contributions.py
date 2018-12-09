from sacremoses import MosesDetokenizer

DETOKENIZER = MosesDetokenizer()

def calculate_contributions(model, original_probabilities, tokens, headline, body_id, true_label_id):
    """
    Calculate the contribution of every word to the true class probability.
    """
    original_probability = original_probabilities[true_label_id]
    probabilities = _probabilities_without_tokens(model, tokens, headline, body_id)

    return [
        (i, original_probability - p[true_label_id])
        for i, p in probabilities
    ]

def _probabilities_without_tokens(model, tokens, headline, body_id):
    return [
        (i, _probabilities_without_token(model, headline, tokens, i))
        for i, w in enumerate(tokens) if len(w) > 3
    ]

def _probabilities_without_token(model, headline, tokens, token_id):
    without_token = [w for i, w in enumerate(tokens) if i != token_id]
    new_body = DETOKENIZER.detokenize(without_token)
    return model.predict_probabilities([headline], [new_body])[0]
