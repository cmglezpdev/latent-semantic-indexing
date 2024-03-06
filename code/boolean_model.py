import spacy
import os
import pickle
from gensim import corpora
from gensim.models import TfidfModel
from sympy import symbols, to_dnf, sympify
Puedes realizar la conversión de las palabras clave 'and', 'or' y 'not' a los operadores correspondientes '&' (AND), '|' (OR) y '~' (NOT) en la cadena de consulta antes de parsearla con `sympy`. Aquí tienes un ejemplo de cómo hacerlo:
from sympy.logic.boolalg import to_cnf
from sympy.parsing.sympy_parser import parse_expr

def convertir_a_fnc(query_string):
    # Reemplaza las palabras clave por los operadores correspondientes
    query_string = query_string.replace('and', '&').replace('or', '|').replace('not', '~')

    # Parsea la cadena de consulta a una expresión simbólica
    query_expr = parse_expr(query_string)

    # Convierte la expresión en FNC
    fnc_query = to_cnf(query_expr)

    return fnc_query



def get_matching_docs(dictionary, vectorized, query, alpha=0.5):
    modelo_pesos_documentos = [
        {term: freq for term, freq in doc} for doc in corpus_tfidf
    ]


    processed_query = convertir_a_fnc(query)

    resultado = set()
    for termino, documentos in dictionary.items():
        if query_expr.subs({simbolo: (termino in documentos) for simbolo in processed_query.free_symbols}):
            resultado.update(documentos)

    return resultado
