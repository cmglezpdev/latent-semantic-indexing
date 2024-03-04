from query_builder import *
from kulback_leibler_divergence import *
from dataset_processing_LSI import load_data, get_corpus_text
import numpy as np
import gradio as gr
from boolean_model import *


# representacion de los documentos en el espacio semantico latente
#

U, S, Vt, doc_representation, vectorized, dictionary, tokenized_documents = load_data()

load_tf_idf_model(vectorized)


def documents_retrieveral_LSI(query: str):
    """
    Makes the complete process of information documents_retrieveral using LSI

    :param query: query to be processed
    :return: list of documents
    """

    if query == "" or query is None:
        return []
    global U, S, Vt, doc_representation, vectorized, dictionary, tokenized_documents

    processed_query = process_query(
        query, vectorized, tokenized_documents, dictionary, S, U, Vt
    )
    weighted_documents = np.dot(Vt.T, processed_query)

    document_norms = np.linalg.norm(doc_representation, axis=1)

    query_vector_norms = np.linalg.norm(processed_query)

    weighted_documents = weighted_documents / (document_norms * query_vector_norms)

    ordered_indexes = np.argsort(weighted_documents)[::-1]

    return [get_corpus_text(i) for i in ordered_indexes[:4]]


def boolean_model_retrieveral(query: str):
    """
    Makes the complete process of information retrieval using boolean model

    :param query: query to be processed
    :return: list of documents
    """
    if query == "" or query is None:
        return []
    global dictionary
    return [
        get_corpus_text(i) for i in get_matching_docs(dictionary, vectorized, query)[:4]
    ]


def search(query1, query2):
    return "\n---------match-------\n\n".join(
        documents_retrieveral_LSI(query1)
    ), "\n-------match-------\n".join(boolean_model_retrieveral(query2))


interface = gr.Interface(
    fn=search,
    inputs=["text", "text"],
    outputs=["text", "text"],
    live=True,
    title="recuperacion con informacion semantica latente",
    description="pon una consulta para buscar en el sistema",
)


# interface.outputs[0].style = "text"
# interface.outputs[0].description = "Documentos mas relevantes"

interface.launch(share=True)
