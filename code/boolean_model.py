import spacy
<<<<<<< HEAD
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
=======
from processing import *
from sympy import sympify, to_dnf

# Descargar modelo de Spacy
nlp = spacy.load("en_core_web_sm")  # loading spacy module

def query_to_dnf(query: str):
    processed_query = query.replace('AND', '&').replace('OR', '|').replace('NOT', '~')
    
    # Convertir a expresión sympy y aplicar to_dnf
    query_expr = sympify(processed_query, evaluate=False)
    query_dnf = to_dnf(query_expr, simplify=True)
    return query_dnf
>>>>>>> 945c07840c239b40c07a8a41034f662a83482400

    # Convierte la expresión en FNC
    fnc_query = to_cnf(query_expr)

<<<<<<< HEAD
    return fnc_query

=======
def split_in_comp(query_dnf):
    comps = str(query_dnf).split('|')
    return [query_to_dnf(c) for c in comps]
>>>>>>> 945c07840c239b40c07a8a41034f662a83482400


def get_matching_docs(dictionary, corp_rep, query_dnf):
    # Función para verificar si un documento satisface una componente conjuntiva de la consulta
    comps = split_in_comp(query_dnf)
    comps = [[" ".join([str(sb) for sb in cmp.free_symbols])] for cmp in comps]
    comps = [doc[0] for doc in comps] # flatten
    comps = tokenize(comps)
    comps = noise_removal(comps)
    comps = stopword_elimination(comps)
    comps = lemmatization(comps)
    
    comps_rep = [list(map(lambda x: x[0], dictionary.doc2bow(comp))) for comp in comps]

<<<<<<< HEAD

    processed_query = convertir_a_fnc(query)

    resultado = set()
    for termino, documentos in dictionary.items():
        if query_expr.subs({simbolo: (termino in documentos) for simbolo in processed_query.free_symbols}):
            resultado.update(documentos)

    return resultado
=======
    matching_documents = []
    for i, doc in enumerate(corp_rep):
        doc_ids = list(map(lambda x: x[0], doc))
        for comp in comps_rep:
            include = True
            for sb in comp:
                if sb not in doc_ids:
                    include = False
                    break
            
            if include:
                matching_documents.append(i)

    return matching_documents
>>>>>>> 945c07840c239b40c07a8a41034f662a83482400
