import pandas as pd
import numpy as np
from typing import List
import re
import string


class Lemmatizer:
    def __init__(self, data: pd.DataFrame):
        data = data.drop_duplicates(subset=["Слово"])
        self.word_map = (data["Исходное слово"] + "=" + data["Часть речи"]).set_axis(
            data["Слово"]
        )
        self.chars_to_remove = "".join(
            set(string.punctuation + '«»"„“‘’()[]{};:+-*/=@#$%^&')
        )

    def _tokenize(self, text: str) -> List[str]:
        pattern = f"[{re.escape(self.chars_to_remove)}]"
        text = re.sub(pattern, "", text).strip()
        return text.split()

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        lemmatized_tokens = []
        for word in tokens:
            word_l = word.lower()
            if word_l in self.word_map.index:
                description = self.word_map[word_l]
            else:
                description = word + "=" + "UNKN"
            lemmatized_tokens.append(f"{word}{{{description}}}")
        return lemmatized_tokens

    def lemmatize_text(self, text: str) -> List[str]:
        tokens = self._tokenize(text)
        return self._lemmatize_tokens(tokens)
