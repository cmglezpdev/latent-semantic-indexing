import spacy
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


def split_in_comp(query_dnf):
    comps = str(query_dnf).split('|')
    return [query_to_dnf(c) for c in comps]


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
