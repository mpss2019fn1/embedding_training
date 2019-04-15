import argparse
import logging

from gensim.models import Word2Vec

from accessible_text_file import AccessibleTextFile
from document_corpus import DocumentCorpus
from tokenization_strategy import BasicTokenizationStrategy


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = _initialize_parser()

    args = parser.parse_args()

    with open("resources/stoplist.txt") as stop_list_file:
        stop_list = stop_list_file.read().split(",")

    tokenization_strategy = BasicTokenizationStrategy()
    tokenization_strategy.set_stop_list(stop_list)

    corpus = DocumentCorpus(args.input, BasicTokenizationStrategy())

    model = Word2Vec(corpus, size=100, window=5, min_count=5, workers=4, sg=0)

    model.wv.save_word2vec_format("wv.model")
    model.save("word2vec.model")


def _initialize_parser():
    general_parser = argparse.ArgumentParser(description='Training word embeddings using gensim')
    general_parser.add_argument("--input", help='Cleaned text corpus', action=AccessibleTextFile, required=True)
    general_parser.add_argument("--output", help='Model containing word embeddings', required=True)

    return general_parser


if __name__ == "__main__":
    main()
