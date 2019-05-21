from data_preparation.abstract_corpus import AbstractCorpus


class WordCorpus(AbstractCorpus):

    def __iter__(self):
        for line in open(self._location):
            # assume there's one document per line, tokens separated by whitespace
            yield self._tokenizer.tokenize(line.replace("\n", ""))
