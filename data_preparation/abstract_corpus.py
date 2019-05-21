from abc import abstractmethod, ABC


class AbstractCorpus(ABC):

    def __init__(self, location, tokenizer):
        self._location = location
        self._tokenizer = tokenizer

    @abstractmethod
    def __iter__(self):
        """Method required"""
