import numpy as np

"""
In this Module We will Use kullback_leibler_divergence to calculate the Kullback-Leibler divergence between two vectors
"""


def kullback_leibler_divergence(vector1, vector2, epsilon=1e-10) -> float:
    """
    :param vector1: first vector
    :param vector2: second vector

    :return float: Kullback-Leibler divergence
    """
    # adds a little value for numeric errors with log
    p = np.asarray(vector1, dtype=np.float64)
    q = np.asarray(vector2, dtype=np.float64)

    p = p + epsilon
    q = q + epsilon

    return np.sum(p * np.log(p / q))
