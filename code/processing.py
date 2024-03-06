import ir_datasets
import gensim
import spacy
import typing

"""
This Module is used to load and process the Corpus 
"""

nlp = spacy.load("en_core_web_sm")  # loading spacy module


def load_corpus() -> list[str]:
    """
    Load the corpus from cranfield dataset

    Returns:
        list[str]: List of documents
    """

    dataset = ir_datasets.load("cranfield")
    corpus = [doc.text for doc in dataset.docs_iter()]
    return corpus


def tokenize( text: list[list[str]] ) -> list[list[spacy.tokens.Token]]:
    """
    Tokenize a list of documents to spacy tokens

    Args:
        text (list[list[str]]): List of documents (each document is a list of sentences)

    Returns:
        list[list[spacy.tokens.Token]]: List of list of spacy tokens for each sentences in each document
    """

    return [[token for token in nlp(document)] for document in text]


def noise_removal( tokens: list[spacy.tokens.Token] ) -> list[spacy.tokens.Token]:
    """
    Remove the noise from the corpus , words such as "the" and "a" are removed
    
    Args:
        tokens (list[spacy.tokens.Token]): The tokenized corpus

    Returns:
        list[spacy.tokens.Token]: Filtered tokens
    """

    return [
        [token for token in tokenized_doc if token.is_alpha] for tokenized_doc in tokens
    ]


def stopword_elimination( tokens: list[spacy.tokens.Token] ) -> list[spacy.tokens.Token]:
    """
    Removes the Stopwords of the corpus
    
    Args:
        tokens (list[spacy.tokens.Token]): Tokenized corpus

    Returns:
        list[spacy.tokens.Token]: Filtered corpus without stopwords
    """

    return [
        [
            token
            for token in tokenized_doc
            if token.text not in spacy.lang.en.stop_words.STOP_WORDS
        ]
        for tokenized_doc in tokens
    ]


def lemmatization( tokens: list[spacy.tokens.Token] ) -> list[str]:
    """
    Lemmantization of the corpus

    Args:
        tokens (list[spacy.tokens.Token]): Tokenized corpus

    Returns:
        list[str]: Corpus with tokens that represent the lemma of each word
    """
    return [[token.lemma_ for token in tokenized_doc] for tokenized_doc in tokens]


def frecuency_filtering( tokens: list[spacy.tokens.Token], no_below: int = 10, no_above: int = 0.5 ) -> list[spacy.tokens.Token]:
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


def build_vocabulary(dictionary: gensim.corpora.Dictionary) -> list[str]:
    """
    Build the Vocabulary from a dictionary containing all the tokens

    :param tokens: List[List[spacy.tokens.Token]] : The tokenized corpus

    :return List[str] : The Vocabulary is built
    """

    return list(dictionary.token2id)


def vector_representation(
    tokens: list[list[spacy.tokens.Token]],
    dictionary: gensim.corpora.Dictionary,
) -> list[typing.Tuple[any, any]]:
    """
    Make the vector representation of the corpus using the Bag of Words Model

    :param tokens: List[List[spacy.tokens.Token]] : The tokenized corpus

    :return List[gensim.corpora.BagOfWords] : The vector representation of the corpus
    """

    return [dictionary.doc2bow(doc) for doc in tokens]


def speech_recognition(
    tokens: list[list[spacy.tokens.Token]],
) -> list[str]:
    """
    Finds the speech parts of the texts
    :param List[List[spacy.tokens.Token]] tokens : The tokenized corpus

    :return List[List[str,str]] :  the speech part identified
    """

    return [
        [(token.text, token.tag_) for token in tokenized_doc]
        for tokenized_doc in tokens
    ]
