import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from Lemmatizer import Lemmatizer
from TaggerHMM import TaggerHMM


class LemmatizerHMM(Lemmatizer):
    def __init__(self, data: pd.DataFrame, train_df: pd.DataFrame):
        super().__init__(data)
        self.word_forms_full: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for _, row in data.iterrows():
            word = str(row["Слово"]).lower()
            lemma = str(row["Исходное слово"])
            pos = str(row["Часть речи"])
            self.word_forms_full[word].append((lemma, pos))
        self.hmm_tagger = TaggerHMM(smoothing_alpha=1.0)
        self.allowed_tags_map: Dict[str, Set[str]] = defaultdict(set)
        for word, forms in self.word_forms_full.items():
            for _, pos in forms:
                self.allowed_tags_map[word].add(pos)
        self.hmm_tagger.fit(train_df)

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        predicted_tags = self.hmm_tagger.viterbi(tokens, self.allowed_tags_map)
        res = []
        for i, word in enumerate(tokens):
            w_lower = word.lower()
            tag = predicted_tags[i]
            forms = self.word_forms_full.get(w_lower, [])
            if forms:
                matched = [lemma for lemma, pos in forms if pos == tag]
                if matched:
                    lemma_pos_str = f"{matched[0]}={tag}"
                else:
                    first_lemma, first_pos = forms[0]
                    lemma_pos_str = f"{first_lemma}={first_pos}"
            else:
                lemma_pos_str = f"{word}={tag}"
            res.append(f"{word}{{{lemma_pos_str}}}")
        return res
