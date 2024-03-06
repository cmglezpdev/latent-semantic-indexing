import typing
import numpy as np
import gensim.corpora as corpora
from gensim.models import LsiModel
from gensim import matutils

"""
This Module is for Singular Value Decomposition
"""


def svd_descomposition( corpus: typing.List[typing.Tuple[int, float]], vocabulary: typing.List[str] ):
    """
    SVD descomposition

    Args:
        corpus (typing.List[typing.Tuple[int, float]]): BagOfWord corpus representing document term descomposition
        vocabulary (typing.List[str]): Dictionary with the vocaulary

    Returns:
        _type_: Descomosition in SVD
    """
    num_docs = len(corpus)
    num_term = len(vocabulary)

    term_doc_matrix = np.zeros((num_term, num_docs))

    # creating the document-term matrix
    for doc_id, doc_bow in enumerate(corpus):
        for term_id, term_freq in doc_bow:
            term_doc_matrix[term_id, doc_id] = term_freq

    U, S, Vt = np.linalg.svd(term_doc_matrix, full_matrices=False)

    # selecting the ammount of singular terms
    k = 500

    # fitting the dimentions for the original matrix
    U_reduced = U[:, :k]
    S_reduced = np.diag(S[:k])
    Vt_reduced = Vt[:k, :]

    doc_representation = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))

    return U_reduced, S_reduced, Vt_reduced, doc_representation.T
