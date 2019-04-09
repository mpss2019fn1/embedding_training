class DocumentCorpus(object):

    def __init__(self, text_file, tokenization):
        self._file = text_file
        self._tokenizationStrategy = tokenization

    def __iter__(self):
        for line in open(self._file):
            # assume there's one document per line, tokens separated by whitespace
            yield self._tokenizationStrategy.tokenize(line.replace("\n", ""))
