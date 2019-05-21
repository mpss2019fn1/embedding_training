import re

from data_preparation.abstract_tokenizer import AbstractTokenizer


class SimpleRegexTokenizer(AbstractTokenizer):

    def __init__(self, stop_words):
        super().__init__(stop_words)
        # Match words ignoring special characters at the beginning or end
        self._token_pattern = re.compile(r"^[^a-zA-Z0-9']?(?P<word>[a-zA-Z0-9',._\"-]+?)[^a-zA-Z0-9']?$")

    def tokenize(self, line):
        lower_tokens = line.lower().split(" ")
        filtered_tokens = (token for token in lower_tokens if token not in self._stop_words)
        cleaned_tokens = (self._clean_token(token) for token in filtered_tokens)
        return [token for token in cleaned_tokens if token]

    def _clean_token(self, token):
        match = self._token_pattern.match(token)
        if not match:
            return ""
        return match.group('word')
