import spacy
import os
import pickle
from gensim import corpora
from gensim.models import TfidfModel
from sympy import symbols, to_dnf, sympify
from processing import *

# Descargar modelo de Spacy
nlp = spacy.load("en_core_web_sm")  # loading spacy module
# Documentos de ejemplo
documentos = [
    "Este es el primer documento.",
    "Este documento es el segundo documento.",
    "Y este es el tercer documento.",
]

modelo_tfidf = None
corpus_tfidf = None


def load_tf_idf_model(corpus):
    global modelo_tfidf, corpus_tfidf

    if os.path.exists("./data/tf_idf_model.pkl"):
        modelo_tfidf = pickle.load(open("../data/tf_idf_model.pkl", "rb"))
        corpus_tfidf = pickle.load(open("../data/tf_idf_corpus.pkl", "rb"))
    else:
        modelo_tfidf = TfidfModel(corpus)
        corpus_tfidf = modelo_tfidf[corpus]
        pickle.dump(modelo_tfidf, open("./data/tf_idf_model.pkl", "wb"))
        pickle.dump(corpus_tfidf, open("./data/tf_idf_corpus.pkl", "wb"))


def get_matching_docs(dictionary, vectorized, query, alpha=0.5):
    modelo_pesos_documentos = [
        {term: freq for term, freq in doc} for doc in corpus_tfidf
    ]

    query_tokens = tokenize([query])

    query_tokens = noise_removal(query_tokens)
    query_tokens = stopword_elimination(query_tokens)
    query_tokens = lemmatization(query_tokens)
    consulta_tfidf = modelo_tfidf[dictionary.doc2bow(query_tokens[0])]

    documentos_relevantes = [
        i
        for i, pesos_documento in enumerate(modelo_pesos_documentos)
        if calcular_similitud(pesos_documento, consulta_tfidf) >= alpha
    ]

    return documentos_relevantes


# Función para calcular la similitud
def calcular_similitud(pesos_documento, consulta_tfidf):
    similitud = sum(
        pesos_documento.get(term, 0) * peso_consulta
        for term, peso_consulta in consulta_tfidf
    )
    return similitud
