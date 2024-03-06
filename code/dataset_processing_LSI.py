from processing import *
from SVD_descomposition import *
import pickle
import os
from constants import DATA_SAVE_DIR

corpus = []


def process_data():
    """
    Loads And process the corpus data

    :return U, S, Vt, doc_representation, vectorized, dictionary : data from pre processing task
    """
    global corpus

    corpus = load_corpus()
    tokenized_documents = tokenize(corpus)
    tokenized_documents = noise_removal(tokenized_documents)
    tokenized_documents = stopword_elimination(tokenized_documents)
    tokenized_documents = lemmatization(tokenized_documents)

    tokenized_documents, dictionary = frecuency_filtering(tokenized_documents, 0, 0.5)

    vocabulary = build_vocabulary(dictionary)

    vectorized = vector_representation(tokenized_documents, dictionary)

    U, S, Vt, doc_representation = svd_descomposition(vectorized, vocabulary)

    pickle.dump(S, open(DATA_SAVE_DIR + "/S.pkl", "wb"))
    pickle.dump(Vt, open(DATA_SAVE_DIR + "/Vt.pkl", "wb"))
    pickle.dump(
        doc_representation, open(DATA_SAVE_DIR + "/doc_representation.pkl", "wb")
    )  # matriz de termino - documento
    pickle.dump(corpus, open(DATA_SAVE_DIR + "/corpus.pkl", "wb"))
    pickle.dump(vectorized, open(DATA_SAVE_DIR + "/vectorized_corpus.pkl", "wb"))
    pickle.dump(dictionary, open(DATA_SAVE_DIR + "/corpus_dictionary.pkl", "wb"))
    pickle.dump(U, open(DATA_SAVE_DIR + "/U.pkl", "wb"))
    pickle.dump(
        tokenized_documents,
        open(DATA_SAVE_DIR + "/tokenized_documents.pkl", "wb"),
    )

    return U, S, Vt, doc_representation, vectorized, dictionary, tokenized_documents


def load_data():
    """
    Checks if there are stored data and decides whether to load or process data
    returns U, S, Vt, doc_representation, vectorized, dictionary data stored or processed
    """
    if any(os.listdir(DATA_SAVE_DIR)):
        print("cargando datos :  .....")
        U, S, Vt, doc_representation, vectorized, dictionary, tokenized_documents = (
            load_from_memory()
        )
    else:
        print("procesando datos:  .....")
        U, S, Vt, doc_representation, vectorized, dictionary, tokenized_documents = (
            process_data()
        )

    return U, S, Vt, doc_representation, vectorized, dictionary, tokenized_documents


def load_from_memory():
    """
    loads the data stored in data_save_directory
    """
    global corpus
    U = pickle.load(open(DATA_SAVE_DIR + "/U.pkl", "rb"))
    S = pickle.load(open(DATA_SAVE_DIR + "/S.pkl", "rb"))
    Vt = pickle.load(open(DATA_SAVE_DIR + "/Vt.pkl", "rb"))
    doc_representation = pickle.load(
        open(DATA_SAVE_DIR + "/doc_representation.pkl", "rb")
    )
    corpus = pickle.load(open(DATA_SAVE_DIR + "/corpus.pkl", "rb"))
    vectorized = pickle.load(open(DATA_SAVE_DIR + "/vectorized_corpus.pkl", "rb"))
    dictionary = pickle.load(open(DATA_SAVE_DIR + "/corpus_dictionary.pkl", "rb"))
    tokenized_documents = pickle.load(
        open(DATA_SAVE_DIR + "/tokenized_documents.pkl", "rb")
    )
    return U, S, Vt, doc_representation, vectorized, dictionary, tokenized_documents


def get_corpus_text(index: int) -> str:
    """
    Retrieves the document text by index
    :param index: index of document
    :return str: document text
    """
    if index < 0:
        return ""
    return corpus[index]
