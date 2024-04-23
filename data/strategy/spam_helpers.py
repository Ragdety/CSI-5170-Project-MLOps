import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text):
    # Remove punctuation and stop words
    punc_rmv = [char for char in text if char not in string.punctuation]
    punc_rmv = "".join(punc_rmv)

    # Remove stop words
    stopword_rmv = [w.strip().lower() for w in punc_rmv.split() if w.strip().lower() not in ENGLISH_STOP_WORDS]

    # Return clean text
    return " ".join(stopword_rmv)