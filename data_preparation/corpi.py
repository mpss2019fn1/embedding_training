from abc import abstractmethod, ABCMeta
from pathlib import Path

from gensim.models.doc2vec import TaggedDocument

from resources import constant


class CorpusAbstract(object):
    __metaclass__ = ABCMeta

    def __init__(self, location, tokenization):
        self._location = location
        self._tokenizationStrategy = tokenization

    @abstractmethod
    def __iter__(self):
        """Method required"""


class WordCorpus(CorpusAbstract):

    def __iter__(self):
        for line in open(self._location):
            # assume there's one document per line, tokens separated by whitespace
            yield self._tokenizationStrategy.tokenize(line.replace("\n", ""))


class DocumentCorpus(CorpusAbstract):

    def __iter__(self):
        dir_path = Path(self._location)

        for file in dir_path.iterdir():
            if file.is_file() and file.name.endswith(constant.WIKIPEDIA_ARTICLE_FILE_ENDING):
                yield from self._process_single_article(file, file.stem)
            else:
                continue

    def _process_single_article(self, article, article_name):
        yield TaggedDocument(self._tokenizationStrategy.tokenize(article.read_text().replace("\n", "")), [article_name])
