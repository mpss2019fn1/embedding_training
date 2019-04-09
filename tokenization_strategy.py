from abc import abstractmethod, ABCMeta
import re


class TokenizationStrategyAbstract(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._stop_list = []

    @abstractmethod
    def tokenize(self, line):
        """Method required"""

    def set_stop_list(self, stop_list):
        self._stop_list = stop_list


class BasicTokenizationStrategy(TokenizationStrategyAbstract):

    def __init__(self):
        super(BasicTokenizationStrategy, self).__init__()
        # Match words ignoring special characters at the beginning or end
        self._token_pattern = re.compile("^[^a-zA-Z0-9']?(?P<word>[a-zA-Z0-9']+?)[^a-zA-Z0-9']?$")

    def tokenize(self, line):
        lower_tokens = line.lower().split(" ")
        filtered_tokens = (token for token in lower_tokens if token not in self._stop_list)
        cleaned_tokens = (self._clean_token(token) for token in filtered_tokens)
        return [token for token in cleaned_tokens if token]

    def _clean_token(self, token):
        match = self._token_pattern.match(token)
        if not match:
            return ""
        return match.group('word')
