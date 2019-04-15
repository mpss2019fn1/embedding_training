import argparse
import logging

from gensim.models import Word2Vec, Doc2Vec

from util.filesystem_validators import AccessibleDirectory
from data_preparation.corpi import DocumentCorpus
from data_preparation.tokenization_strategy import BasicTokenizationStrategy


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = _initialize_parser()

    args = parser.parse_args()

    with open("resources/stoplist.txt") as stop_list_file:
        stop_list = stop_list_file.read().split(",")

    tokenization_strategy = BasicTokenizationStrategy()
    tokenization_strategy.set_stop_list(stop_list)

    corpus = DocumentCorpus(args.input, BasicTokenizationStrategy())

    # model = Word2Vec(corpus, size=100, window=5, min_count=5, workers=4, sg=0)
    model = _train_doc_embeddings(corpus)

    model.wv.save_word2vec_format("doc2vec.text.model")
    model.save("doc2vec.binary.model")


def _train_word_embeddings(corpus):
    return Word2Vec(corpus, size=100, window=5, min_count=5, workers=4, sg=0)


def _train_doc_embeddings(corpus):
    model = Doc2Vec(vector_size=100, min_count=2, epochs=40)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def _initialize_parser():
    general_parser = argparse.ArgumentParser(description='Training word embeddings using gensim')
    # general_parser.add_argument("--input", help='Cleaned text corpus', action=AccessibleTextFile, required=True)
    general_parser.add_argument("--input", help='Directory containing cleaned Wikipedia articles',
                                action=AccessibleDirectory, required=True)
    general_parser.add_argument("--output", help='Model containing word embeddings', required=True)

    return general_parser


if __name__ == "__main__":
    main()
