import numpy as np


def email_features(word_indices, vocabList_d):
    n = len(vocabList_d)
    features = np.zeros((n, 1))

    for i in word_indices:
        features[i] = 1

    return np.array(features)
