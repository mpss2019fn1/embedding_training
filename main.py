import argparse
import logging
import resources

from gensim.models import Word2Vec, Doc2Vec

from pathlib import Path

from data_preparation import SimpleRegexTokenizer, DocumentCorpus
from data_preparation.paragraph_corpus import ParagraphCorpus
from util.filesystem_validators import AccessibleDirectory, AccessibleTextFile


def main():
    logging.basicConfig(format="%(asctime)s : [%(threadName)s] %(levelname)s : %(message)s", level=logging.INFO)
    parser = _initialize_parser()

    args = parser.parse_args()

    with open(args.stop_words, "r") as stop_list_file:
        stop_list = stop_list_file.read().split(",")

    tokenizer = SimpleRegexTokenizer(stop_list)
    corpus = ParagraphCorpus(args.input, tokenizer)

    # model = Word2Vec(corpus, size=100, window=5, min_count=5, workers=4, sg=0)
    model = _train_doc_embeddings(corpus, args.workers, args.epochs)

    if args.output_doc_vecs:
        model.save_word2vec_format(Path(args.output, args.output_doc_vecs), doctag_vec=True,
                                   word_vec=False, prefix="*entity_")

    if args.output_word_vecs:
        model.save_word2vec_format(Path(args.output, args.output_word_vecs), doctag_vec=False,
                                   word_vec=True, prefix="*word_")

    model.save(Path(args.output, resources.constant.BINARY_OUTPUT_FILE))


def _train_word_embeddings(corpus, workers):
    return Word2Vec(corpus, size=100, window=5, min_count=5, workers=workers, sg=0)


def _train_doc_embeddings(corpus, workers, epochs):
    return Doc2Vec(corpus, vector_size=100, min_count=2, epochs=epochs, workers=workers)


def _initialize_parser():
    general_parser = argparse.ArgumentParser(description="Training word embeddings using gensim")
    general_parser.add_argument("--input", help="Directory containing cleaned Wikipedia articles",
                                action=AccessibleDirectory, required=True)
    general_parser.add_argument("--output", help="Desired location for model containing word embeddings", required=True,
                                action=AccessibleDirectory)
    general_parser.add_argument("--workers", help="Workers (e.g. processes) to use for training", required=False,
                                type=int, default=16)
    general_parser.add_argument("--epochs", help="Number of epochs to use for training", required=False, type=int,
                                default=50)
    general_parser.add_argument("--output-doc-vecs", help="File to save the trained document vectors to",
                                required=False, default="doc-vecs.txt")
    general_parser.add_argument("--output-word-vecs", help="File to save the trained word vectors to",
                                required=False, default="word-vecs.txt")
    general_parser.add_argument("--stop-words", help="File, which contains a list of stop words (separated by ',')",
                                required=False, default="resources/stoplist.txt", action=AccessibleTextFile)

    return general_parser


if __name__ == "__main__":
    main()
