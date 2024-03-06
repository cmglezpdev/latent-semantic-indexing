from boolean_model import get_matching_docs
from dataset_processing_LSI import get_corpus_text, load_data
from query_aumentation import *
from processing import *
import typing
import gensim
import spacy
import numpy as np
import pickle
import os
from constants import DATA_SAVE_DIR

nlp = spacy.load("en_core_web_sm")

"""
The mean of this module is to make the processing of the query
"""

U, S, Vt, doc_representation, vectorized, dictionary, tokenized_documents = load_data()

def process_query(
    query: str,
    corpus: typing.List[typing.List[tuple[any, any]]],
    tokenized_documents: typing.List[typing.List[any]],
    corpus_diccionary: gensim.corpora.Dictionary,
    S_reduced: np.array,
    U_reduced: np.array,
    Vt_reduced: np.array,
):
    """
    The Whole Query process flow

    :param query: The query
    :param corpus: The pre processed corpus
    :param tokenized_documents: The tokenized corpus

    :param S_reduced: The S matrix from the SVD descomposition reduced
    :param U_reduced: The U matrix from the SVD descomposition reduced
    :param Vt_reduced: The V matrix from the SVD descomposition reduced
    """

    lda_model = None

    if os.path.exists(f"{DATA_SAVE_DIR}/lda_model.pkl"):
        lda_model = pickle.load(open(f"{DATA_SAVE_DIR}/lda_model.pkl", "rb"))
    else:
        lda_model = training_lda_model(corpus, corpus_diccionary)
        pickle.dump(lda_model, open(f"{DATA_SAVE_DIR}/lda_model.pkl", "wb"))

    query = expand_query(query, lda_model, corpus_diccionary)

    # print(f"Expanded query: {query}")

    query_tokens = tokenize([query])
    query_tokens = noise_removal(query_tokens)
    query_tokens = stopword_elimination(query_tokens)
    query_tokens = lemmatization(query_tokens)
    query_bow = np.zeros(len(corpus_diccionary))

    for term_id, weight in corpus_diccionary.doc2bow(query_tokens[0]):
        query_bow[term_id] = weight

    query_projection_original = np.dot(
        np.linalg.inv(S_reduced), np.dot(U_reduced.T, query_bow)
    )

    return query_projection_original


def documents_retrieveral_LSI(query: str):
    """
    Makes the complete process of information documents_retrieveral using LSI

    :param query: query to be processed
    :return: list of documents
    """

    if query == "" or query is None:
        return []
    global U, S, Vt, doc_representation, vectorized, dictionary, tokenized_documents

    processed_query = process_query(query, vectorized, tokenized_documents, dictionary, S, U, Vt)
    
    # cosine distance
    weighted_documents = np.dot(Vt.T, processed_query)
    document_norms = np.linalg.norm(doc_representation, axis=1)
    query_vector_norms = np.linalg.norm(processed_query)
    weighted_documents = weighted_documents / (document_norms * query_vector_norms)
    
    ordered_indexes = np.argsort(weighted_documents)[::-1]
    # print(ordered_indexes)
    return [(i, get_corpus_text(i)) for i in ordered_indexes[:4]]


def boolean_model_retrieveral(query: str):
    """
    Makes the complete process of information retrieval using boolean model

    :param query: query to be processed
    :return: list of documents
    """
    if query == "" or query is None:
        return []

    global dictionary, vectorized
    try:
        return [(i, get_corpus_text(i)) for i in get_matching_docs(dictionary, vectorized, query)[:4]]
    except Exception as e:
        print(e)
        return [] 
