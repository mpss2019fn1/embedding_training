from pathlib import Path

from gensim.models.doc2vec import TaggedDocument

from data_preparation.abstract_corpus import AbstractCorpus
from resources import constant


class DocumentCorpus(AbstractCorpus):

    def __iter__(self):
        dir_path = Path(self._location)

        for file in dir_path.iterdir():
            if file.is_file() and file.name.endswith(constant.WIKIPEDIA_ARTICLE_FILE_ENDING):
                yield from self._process_single_article(file, file.stem)
            else:
                continue

    def _process_single_article(self, article, article_name):
        yield TaggedDocument(self._tokenizer.tokenize(article.read_text().replace("\n", "")), [article_name])
