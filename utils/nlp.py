from nltk import pos_tag, pos_tag_sents, word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn

def find_synonym(token, pos):
    synsets = wn.synsets(token, pos=pos)

    if synsets:
        # Use the first synset.
        synonyms = synsets[0].lemma_names()

        # Find the first word that is not our token.
        for synonym in synonyms:
            if synonym != token:
                return _remove_underscores(synonym)

def tokenize(body):
    sentence_tokens = [word_tokenize(sentence) for sentence in sent_tokenize(body)]
    return [w for sentence in sentence_tokens for w in sentence]

def tokenize_and_tag(body):
    sentence_tokens = [word_tokenize(sentence) for sentence in sent_tokenize(body)]
    tagged_sentences = pos_tag_sents(sentence_tokens)
    flattened = [tagged for sentence in tagged_sentences for tagged in sentence]
    return [(token, _translate_tag(tag)) for token, tag in flattened]

def _remove_underscores(word):
    return ' '.join(word.split('_'))

def _translate_tag(penntag):
    """ The tagger uses different tags than wordnet. """
    map = {
        'NN':wn.NOUN,
        'JJ':wn.ADJ,
        'VB':wn.VERB,
        'RB':wn.ADV
    }

    return map.get(penntag[:2], wn.NOUN) # Use first 2 chars, and fall back to noun.
