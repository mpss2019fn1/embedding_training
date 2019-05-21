from pathlib import Path

from gensim.models.doc2vec import TaggedDocument

from data_preparation import AbstractCorpus
from resources import constant


class ParagraphCorpus(AbstractCorpus):

    def __iter__(self):
        dir_path = Path(self._location)

        for file in dir_path.iterdir():
            if file.is_file() and file.name.endswith(constant.WIKIPEDIA_ARTICLE_FILE_ENDING):
                yield from self._process_single_article(file, file.stem)
            else:
                continue

    def _process_single_article(self, article, article_name):
        with article.open("r") as source:
            for index, line in enumerate(source, 1):
                yield TaggedDocument(self._tokenizer.tokenize(line.replace("\n", "")),
                                     [article_name, f"{article_name}_{index}"])
