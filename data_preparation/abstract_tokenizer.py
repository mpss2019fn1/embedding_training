from abc import ABC, abstractmethod


class AbstractTokenizer(ABC):

    def __init__(self, stop_words):
        self._stop_words = stop_words

    @abstractmethod
    def tokenize(self, line):
        raise NotImplementedError()
