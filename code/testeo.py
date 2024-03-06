from metrics import evaluate_model
from query_builder import *
from kulback_leibler_divergence import *
from dataset_processing_LSI import load_data, get_corpus_text
import numpy as np
import gradio as gr
from boolean_model import *


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
    print(ordered_indexes)
    return [(i, get_corpus_text(i)) for i in ordered_indexes[:4]]



x = [0, 0, 0, 1, 0, 0, 0]

print(1 in x)



# x = evaluate_model(documents_retrieveral_LSI)

# print(x)