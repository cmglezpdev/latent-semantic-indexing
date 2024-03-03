import ir_datasets
import gensim
import spacy
import typing
import pickle


"""
This Module is used to load and process the Corpus 
"""

nlp = spacy.load("en_core_web_sm")  # loading spacy module


def load_corpus() -> typing.List[str]:
    """
    Load the corpus

    return List[str]
    """
    dataset = ir_datasets.load("cranfield")

    #    queries = []
    #    for query in dataset.queries_iter():
    #        queries.append(query.text)  # query.text
    #        print(query)
    #        pickle.dump(query.text, open("query.txt", "wb"))
    #
    #    pickle.dump(queries, open("querys.txt", "wb"))

    corpus = [doc.text for doc in dataset.docs_iter()]
    return corpus


def tokenize(
    text: typing.List[typing.List[str]],
) -> typing.List[typing.List[spacy.tokens.Token]]:
    """
    Tonekize the dataset
    :param List[List[str]] text : The corpus content


    :return List[List[spacy.tokens.Token]] : The tokenized corpus
    """

    return [[token for token in nlp(document)] for document in text]


def noise_removal(
    tokens: typing.List[spacy.tokens.Token],
) -> typing.List[spacy.tokens.Token]:
    """
    Remove the noise from the corpus , words such as "the" and "a" are removed
    :param List[spacy.tokens.Token] tokens : The tokenized corpus

    :return List[spacy.tokens.Token] : The noise removed corpus
    """

    return [
        [token for token in tokenized_doc if token.is_alpha] for tokenized_doc in tokens
    ]


def stopword_elimination(
    tokens: typing.List[spacy.tokens.Token],
) -> typing.List[spacy.tokens.Token]:
    """
    Removes the Stopwords from the corpus
    :param List[spacy.tokens.Token] tokens : The tokenized corpus

    :return List[spacy.tokens.Token] : The stopwords removed corpus
    """
    return [
        [
            token
            for token in tokenized_doc
            if token.text not in spacy.lang.en.stop_words.STOP_WORDS
        ]
        for tokenized_doc in tokens
    ]


def lemmatization(
    tokens: typing.List[spacy.tokens.Token],
) -> typing.List[spacy.tokens.Token]:
    """
    :param List[spacy.tokens.Token] tokens : The tokenized corpus

    :return List[spacy.tokens.Token] : The lemmatized corpus
    """
    return [[token.lemma_ for token in tokenized_doc] for tokenized_doc in tokens]


def frecuency_filtering(
    tokens: typing.List[spacy.tokens.Token],
    no_below: int = 10,
    no_above: int = 0.5,
) -> typing.List[spacy.tokens.Token]:
    """
    Filter the tokens by its ocurrencies and returns the tokens and a dictionary with all document tokens
    :param tokens: List[spacy.tokens.Token] : The tokenized corpus
    :param no_below: int : The minimum frequency
    :param no_above: int : The maximum frequency

    :return List[List[spacy.tokens.Token]] : The filtered tokens
    :return gensim.corpora.Dictionary : The filtered tokens

    """
    # crea un diccionario con los tokens de todo el corpus
    dictionary = gensim.corpora.Dictionary(tokens)
    # filtra por los limites especificados
    dictionary.filter_extremes(no_below, no_above)

    # guarda las palabras filtradas
    filtered_words = [word for _, word in dictionary.iteritems()]
    # crea un array de array de tokens donde estan los tokens filtrados por ocurrencias
    filtered_tokens = [
        [word for word in doc if word in filtered_words] for doc in tokens
    ]

    return filtered_tokens, dictionary


def build_vocabulary(dictionary: gensim.corpora.Dictionary) -> typing.List[str]:
    """
    Build the Vocabulary from a dictionary containing all the tokens

    :param tokens: List[List[spacy.tokens.Token]] : The tokenized corpus

    :return List[str] : The Vocabulary is built
    """

    return list(dictionary.token2id)


def vector_representation(
    tokens: typing.List[typing.List[spacy.tokens.Token]],
    dictionary: gensim.corpora.Dictionary,
) -> typing.List[typing.Tuple[any, any]]:
    """
    Make the vector representation of the corpus using the Bag of Words Model

    :param tokens: List[List[spacy.tokens.Token]] : The tokenized corpus

    :return List[gensim.corpora.BagOfWords] : The vector representation of the corpus
    """

    return [dictionary.doc2bow(doc) for doc in tokens]


def speech_recognition(
    tokens: typing.List[typing.List[spacy.tokens.Token]],
) -> typing.List[str]:
    """
    Finds the speech parts of the texts
    :param List[List[spacy.tokens.Token]] tokens : The tokenized corpus

    :return List[List[str,str]] :  the speech part identified
    """

    return [
        [(token.text, token.tag_) for token in tokenized_doc]
        for tokenized_doc in tokens
    ]
