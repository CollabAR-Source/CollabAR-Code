import numpy as np

def nomalize_entropy(probability):
    entropy = 0
    for i in range(probability.shape[1]):
        entropy = (-probability[0][i]*np.log(probability[0][i])) + entropy
    return entropy/np.log(probability.shape[1])