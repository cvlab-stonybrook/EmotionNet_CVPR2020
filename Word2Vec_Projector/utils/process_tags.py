# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 28/Nov/2018 10:33

from collections import Counter
from tqdm import tqdm
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
import nltk



def _count_unique_tokens(tokenized_docs):
    tokens = []
    for i, doc in tokenized_docs:
        tokens += doc
    return Counter(tokens)


def _encode(tokenized_docs, encoder):
    return [(i, [encoder[t] for t in doc]) for i, doc in tokenized_docs]


def _remove_tokens(tokenized_docs, counts, min_counts, max_counts):
    """
    Words with count < min_counts or count > max_counts
    will be removed.
    """
    total_tokens_count = sum(
        count for token, count in counts.most_common()
    )
    print('total number of tokens:', total_tokens_count)
    keep = {}

    if max_counts is not None:
        unknown_tokens_count = sum(
            count for token, count in counts.most_common()
            if count < min_counts or count > max_counts
        )
        print('number of tokens to be removed:', unknown_tokens_count)
        for token, count in counts.most_common():
            keep[token] = count >= min_counts and count <= max_counts
    else:
        unknown_tokens_count = sum(
            count for token, count in counts.most_common()
            if count < min_counts
        )
        print('number of tokens to be removed:', unknown_tokens_count)
        for token, count in counts.most_common():
            keep[token] = count >= min_counts

    return [(i, [t for t in doc if keep[t]]) for i, doc in tokenized_docs]


def _create_token_encoder(counts):

    total_tokens_count = sum(
        count for token, count in counts.most_common()
    )
    print('total number of tokens:', total_tokens_count)

    encoder = {}
    decoder = {}
    word_counts = []
    i = 0

    for token, count in counts.most_common():
        # counts.most_common() is in decreasing count order
        encoder[token] = i
        decoder[i] = token
        word_counts.append(count)
        i += 1

    return encoder, decoder, word_counts


def preprocess_tags(docs, pdefined_vocab=None, min_length=10, min_counts=100, max_counts=None):
    """Tokenize, clean, and encode documents.

    Arguments:
        docs: A list of tuples (index, string), each string is a document.
        nlp: A spaCy object, like nlp = spacy.load('en').
        min_length: An integer, minimum document length.
        min_counts: An integer, minimum count of a word.
        max_counts: An integer, maximum count of a word.

    Returns:
        encoded_docs: A list of tuples (index, list), each list is a document
            with words encoded by integer values.
        word_decoder: A dict, integer -> word.
        word_counts: A list of integers, counts of words that are in word_decoder.
            word_counts[i] is the number of occurrences of word word_decoder[i]
            in all documents in docs.
    """

    def filter_out(doc, pdefined_vocab):
        # text = ' '.join(doc.split())  # remove excessive spaces
        # text = nltk.word_tokenize(text)
        # text = nlp(text, tag=True, parse=False, entity=False)
        # text = nlp(text)

        return [t.lower() for t in doc
                if t in pdefined_vocab and len(t)>2]

    def filter_in(doc, pdefined_vocab):
        return [t.lower() for t in doc
                if t not in pdefined_vocab and len(t)>2]

    if pdefined_vocab is None:
        pdefined_vocab = stopwords
        tokenized_docs = [(i, filter_in(doc, pdefined_vocab)) for i, doc in tqdm(docs)]
    else:
        tokenized_docs = [(i, filter_out(doc, pdefined_vocab)) for i, doc in tqdm(docs)]

    # remove short documents
    n_short_docs = sum(1 for i, doc in tokenized_docs if len(doc) < min_length)
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs if len(doc) >= min_length]
    print('number of removed short documents:', n_short_docs)

    # remove some tokens
    counts = _count_unique_tokens(tokenized_docs)
    tokenized_docs = _remove_tokens(tokenized_docs, counts, min_counts, max_counts)
    n_short_docs = sum(1 for i, doc in tokenized_docs if len(doc) < min_length)
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs if len(doc) >= min_length]
    print('number of additionally removed short documents:', n_short_docs)

    counts = _count_unique_tokens(tokenized_docs)
    word_encoder, word_decoder, word_counts = _create_token_encoder(counts)

    print('\nminimum word count number:', word_counts[-1])
    print('this number can be less than MIN_COUNTS because of document removal')

    encoded_docs = _encode(tokenized_docs, word_encoder)
    return encoded_docs, word_encoder, word_decoder, word_counts