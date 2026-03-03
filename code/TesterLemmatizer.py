import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict
import re


class TesterLemmatizer:
    def __init__(self, lemmatizer):
        self.lemmatizer = lemmatizer
        self.matrix_data = defaultdict(lambda: defaultdict(int))
        self.all_tags = set()
        self.errors = []
        self.correct_lemma_and_pos = 0
        self.correct_lemma = 0
        self.correct_pos = 0
        self.total_words = 0

    def _update_tags_and_matrix(self, true_pos: str, pred_pos: str):
        self.matrix_data[true_pos][pred_pos] += 1
        self.all_tags.update([true_pos, pred_pos])

    def _check_coincidence(self, prediction, answer: Dict):
        pattern = re.compile(r"(.+)\{(.+)=([^}]+)\}")
        prediction = pattern.match(prediction)
        predicted_word, predicted_lemma, predicted_pos = prediction.groups()
        answer_word, answer_lemma, answer_pos = (
            answer["Слово"],
            answer["Исходное слово"],
            answer["Часть речи"],
        )
        self.correct_lemma += answer_lemma == predicted_lemma
        self.correct_pos += answer_pos == predicted_pos
        is_lemma_and_pos_correct = (answer_lemma == predicted_lemma) and (
            answer_pos == predicted_pos
        )
        self.correct_lemma_and_pos += is_lemma_and_pos_correct
        if not is_lemma_and_pos_correct:
            self.errors.append(
                {
                    "word": answer_word,
                    "true_lemma": answer_lemma,
                    "pred_lemma": predicted_lemma,
                    "true_pos": answer_pos,
                    "pred_pos": predicted_pos,
                }
            )
        self._update_tags_and_matrix(answer_pos, predicted_pos)

    def _create_confusion_matrix(self) -> pd.DataFrame:
        tags = sorted(list(self.all_tags))
        if not tags:
            return pd.DataFrame()
        conf_matrix = pd.DataFrame(0, index=tags, columns=tags, dtype=int)
        for t_actual in tags:
            for t_pred in tags:
                conf_matrix.loc[t_actual, t_pred] = self.matrix_data[t_actual][t_pred]
        return conf_matrix

    def evaluate(self, test_df):
        for _, row in test_df.iterrows():
            predictions = self.lemmatizer.lemmatize_text(row["Исходный текст"])
            ans_tokens = row["токены"]
            for i, answer in enumerate(ans_tokens):
                if i == len(predictions):
                    break
                self.total_words += 1
                self._check_coincidence(predictions[i], answer)
        self._print_results()

    def _print_results(self) -> None:
        if self.total_words > 0:
            print("МЕТРИКИ КАЧЕСТВА")
            print("=" * 60)
            print(
                f"Точность (лемма+часть речи):   {self.correct_lemma_and_pos / self.total_words:6.2%}"
            )
            print(f"Точность по леммам:  {self.correct_lemma / self.total_words:6.2%}")
            print(
                f"Точность по частям речи:    {self.correct_pos / self.total_words:6.2%}"
            )
            print(f"Ошибок:           {self.total_words - self.correct_lemma_and_pos}")
