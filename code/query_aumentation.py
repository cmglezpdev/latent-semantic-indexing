import gensim.corpora as corpora
from gensim.models import LdaModel
import typing
from processing import *


"""
In this Module we apply techniques to expand the query in order to increase the prformance of the retrieveral 
In this case we use Semantic Expansion with Latent themes Model
"""


def training_lda_model(
    corpus: typing.List[typing.List[tuple[any, any]]],
    dictionary: corpora.Dictionary,
) -> LdaModel:
    """
    training the model using Latent Drichlet Aloccation

    :param List[BagOfWords] corpus: the vectorized corpus
    :param dictionary corpora.Dictionary: the dictionary of the corpus

    :return LdaModel : the model trained to find the topics
    """
    Lda = LdaModel(
        corpus=corpus,
        num_topics=10,
        id2word=dictionary,
        passes=100,
    )
    return Lda


def expand_query(
    query: str, lda_model: LdaModel, dictionary: corpora.Dictionary
) -> str:
    """
    Expanding query using the corpus topics with lda model

    :param query str: the query provided by the user
    :param LdaModel lda_model: the model trained to find the topics
    :param dictionary corpora.Dictionary: the dictionary of the corpus

    :return str: the expanded query
    """

    query_tokens = tokenize([query])
    query_tokens = noise_removal(query_tokens)
    query_tokens = stopword_elimination(query_tokens)
    query_tokens = lemmatization(query_tokens)
    query_tokens, query_dict = frecuency_filtering(query_tokens, 0, 200)

    vectorized_query = vector_representation(query_tokens, dictionary)

    query_topics = lda_model[vectorized_query[0]]

    expanded_query_terms = []

    for topic_id, weight in query_topics:
        for term, _ in lda_model.show_topic(topic_id, topn=5):
            if weight > 0.5:
                expanded_query_terms.append(term)

    expanded_query = query.split() + expanded_query_terms

    return " ".join(expanded_query)
