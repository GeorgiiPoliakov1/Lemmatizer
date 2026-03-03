import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set
from collections import defaultdict


class TaggerHMM:
    def __init__(self, smoothing_alpha: float = 1.0):
        self.alpha = smoothing_alpha
        self.states: List[str] = []
        self.start_prob: Dict[str, float] = {}
        self.trans_prob: Dict[str, Dict[str, float]] = {}
        self.emiss_prob: Dict[str, Dict[str, float]] = {}
        self.word_to_pos_freq: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def fit(self, train_df: pd.DataFrame):
        sentences = train_df["токены"].tolist()
        trans_counts = defaultdict(lambda: defaultdict(int))
        emiss_counts = defaultdict(lambda: defaultdict(int))
        start_counts = defaultdict(int)
        all_words = set()
        all_pos = set()
        for sentence in sentences:
            if not sentence:
                continue
            tokens = []
            words = []
            for t in sentence:
                pos = t.get("Часть речи", "UNKN")
                word = t.get("Слово", "").lower()
                tokens.append(pos)
                words.append(word)
                all_pos.add(pos)
                all_words.add(word)
                self.word_to_pos_freq[word][pos] += 1
            if not tokens:
                continue
            start_counts[tokens[0]] += 1
            for i in range(len(tokens)):
                if i > 0:
                    trans_counts[tokens[i - 1]][tokens[i]] += 1
                emiss_counts[tokens[i]][words[i]] += 1
        self.states = sorted(list(all_pos))
        n_states = len(self.states)
        n_words = len(all_words)
        total_starts = sum(start_counts.values()) + n_states * self.alpha
        for pos in self.states:
            count = start_counts.get(pos, 0) + self.alpha
            self.start_prob[pos] = np.log(count / total_starts)
        for prev_pos in self.states:
            total_trans = sum(trans_counts[prev_pos].values()) + n_states * self.alpha
            self.trans_prob[prev_pos] = {}
            for curr_pos in self.states:
                count = trans_counts[prev_pos].get(curr_pos, 0) + self.alpha
                self.trans_prob[prev_pos][curr_pos] = np.log(count / total_trans)
        for pos in self.states:
            total_emiss = sum(emiss_counts[pos].values()) + n_words * self.alpha
            self.emiss_prob[pos] = {}
            for word in all_words:
                count = emiss_counts[pos].get(word, 0) + self.alpha
                self.emiss_prob[pos][word] = np.log(count / total_emiss)

    def _get_emission_prob(self, pos: str, word: str) -> float:
        return self.emiss_prob.get(pos, {}).get(word, -10.0)

    def viterbi(
        self,
        observation: List[str],
        allowed_tags_map: Optional[Dict[str, Set[str]]] = None,
    ) -> List[str]:
        if not observation:
            return []
        T = len(observation)
        states = self.states
        N = len(states)
        if N == 0:
            return ["UNKN"] * T
        viterbi = np.full((T, N), -np.inf)
        backpointer = np.zeros((T, N), dtype=int)
        word_0 = observation[0].lower()
        allowed_0 = (
            allowed_tags_map.get(word_0, set(states))
            if allowed_tags_map
            else set(states)
        )
        for s_idx, s in enumerate(states):
            if s not in allowed_0:
                continue
            start_p = self.start_prob.get(s, -10.0)
            emit_p = self._get_emission_prob(s, word_0)
            viterbi[0][s_idx] = start_p + emit_p
        for t in range(1, T):
            word_t = observation[t].lower()
            allowed_t = (
                allowed_tags_map.get(word_t, set(states))
                if allowed_tags_map
                else set(states)
            )

            for s_idx, s in enumerate(states):
                if s not in allowed_t:
                    continue

                emit_p = self._get_emission_prob(s, word_t)
                best_prev_val = -np.inf
                best_prev_idx = 0

                for prev_idx, prev_s in enumerate(states):
                    if viterbi[t - 1][prev_idx] == -np.inf:
                        continue
                    trans_p = self.trans_prob.get(prev_s, {}).get(s, -10.0)
                    val = viterbi[t - 1][prev_idx] + trans_p
                    if val > best_prev_val:
                        best_prev_val = val
                        best_prev_idx = prev_idx

                viterbi[t][s_idx] = best_prev_val + emit_p
                backpointer[t][s_idx] = best_prev_idx
        path = [0] * T
        last_row = viterbi[T - 1]
        if np.all(last_row == -np.inf):
            return ["UNKN"] * T
        path[T - 1] = np.argmax(last_row)
        for t in range(T - 2, -1, -1):
            path[t] = backpointer[t + 1][path[t + 1]]

        return [states[idx] for idx in path]
