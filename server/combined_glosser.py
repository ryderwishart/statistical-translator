from collections import defaultdict, Counter
import re
import math
import numpy as np
from typing import List, Tuple, Dict, Set

class CombinedGlosser:
    def __init__(self, corpus, window_size=3, overlap=0.5, n=3):
        self.corpus = corpus
        self.window_size = window_size
        self.overlap = overlap
        self.n = n
        self.co_occurrences = defaultdict(lambda: defaultdict(int))
        self.source_counts = defaultdict(int)
        self.target_counts = defaultdict(int)
        self.source_doc_freq = defaultdict(int)
        self.target_doc_freq = defaultdict(int)
        self.total_docs = 0
        self.stop_words: Set[str] = set()
        self.sub_models = self.train_sub_models()

    def train_sub_models(self):
        if self.corpus is None:
            return []
        sub_models = []
        slice_size = len(self.corpus) // 5  # Create 5 sub-models
        
        for i in range(5):
            start = i * slice_size
            end = (i + 1) * slice_size
            sub_corpus = self.corpus[start:end]
            sub_model = ImprovedStatisticalGlosser(self.n)
            sub_model.train([s[0] for s in sub_corpus], [s[1] for s in sub_corpus])
            sub_models.append(sub_model)
        
        return sub_models

    def sliding_window_predictions(self, source, target):
        source_tokens = self.tokenize(source)
        target_tokens = self.tokenize(target)
        
        window_predictions = []
        for i in range(0, len(source_tokens) - self.window_size + 1, max(1, int(self.window_size * (1 - self.overlap)))):
            source_window = ' '.join(source_tokens[i:i+self.window_size])
            target_window = ' '.join(target_tokens[i:i+self.window_size])
            
            for sub_model in self.sub_models:
                window_predictions.append(sub_model.gloss(source_window, target_window))
        
        return window_predictions

    def ensemble_statistics(self, window_predictions):
        ensemble_pred = defaultdict(lambda: defaultdict(float))
        
        for predictions in window_predictions:
            for s_ngram, t_mappings in predictions:
                for t_ngram, score in t_mappings:
                    ensemble_pred[s_ngram][t_ngram] += score
        
        # Normalize scores
        for s_ngram in ensemble_pred:
            total = sum(ensemble_pred[s_ngram].values())
            for t_ngram in ensemble_pred[s_ngram]:
                ensemble_pred[s_ngram][t_ngram] /= total
        
        return ensemble_pred

    def mcmc_refinement(self, ensemble_predictions, iterations=1000):
        current_alignment = {s: max(t.items(), key=lambda x: x[1])[0] for s, t in ensemble_predictions.items()}
        current_score = self.alignment_score(current_alignment, ensemble_predictions)
        
        for _ in range(iterations):
            new_alignment = current_alignment.copy()
            s = np.random.choice(list(new_alignment.keys()))
            new_alignment[s] = np.random.choice(list(ensemble_predictions[s].keys()))
            
            new_score = self.alignment_score(new_alignment, ensemble_predictions)
            
            if new_score > current_score or np.random.random() < np.exp(new_score - current_score):
                current_alignment = new_alignment
                current_score = new_score
        
        return current_alignment

    def alignment_score(self, alignment, ensemble_predictions):
        return sum(ensemble_predictions[s][t] for s, t in alignment.items())

    def confidence_scoring(self, alignments):
        scores = {}
        for s, t in alignments.items():
            alternatives = sorted(self.ensemble_predictions[s].items(), key=lambda x: x[1], reverse=True)
            if len(alternatives) > 1:
                score = (alternatives[0][1] - alternatives[1][1]) / alternatives[0][1]
            else:
                score = 1.0
            scores[s] = score
        return scores

    def multi_ngram_analysis(self, alignments):
        multi_ngram_alignments = {}
        for n in range(1, self.n + 1):
            sub_model = ImprovedStatisticalGlosser(n)
            sub_model.train([s[0] for s in self.corpus], [s[1] for s in self.corpus])
            multi_ngram_alignments[n] = sub_model.gloss(' '.join(alignments.keys()), ' '.join(alignments.values()))
        return multi_ngram_alignments

    def divide_and_conquer_gloss(self, source, target):
        tokens_s = self.tokenize(source)
        tokens_t = self.tokenize(target)
        final_glosses = {}
        
        while tokens_s and tokens_t:
            alignments = self.mcmc_refinement(self.ensemble_predictions)
            confidence_scores = self.confidence_scoring(alignments)
            
            best_alignment = max(confidence_scores, key=confidence_scores.get)
            final_glosses[best_alignment] = alignments[best_alignment]
            
            tokens_s = [t for t in tokens_s if t not in best_alignment.split()]
            tokens_t = [t for t in tokens_t if t not in alignments[best_alignment].split()]
            
            # Recalculate ensemble predictions for remaining tokens
            self.ensemble_predictions = self.ensemble_statistics(
                self.sliding_window_predictions(' '.join(tokens_s), ' '.join(tokens_t))
            )
        
        return final_glosses

    def gloss(self, source, target):
        self.window_predictions = self.sliding_window_predictions(source, target)
        self.ensemble_predictions = self.ensemble_statistics(self.window_predictions)
        refined_alignments = self.mcmc_refinement(self.ensemble_predictions)
        scored_alignments = self.confidence_scoring(refined_alignments)
        multi_ngram_alignments = self.multi_ngram_analysis(refined_alignments)
        final_glosses = self.divide_and_conquer_gloss(source, target)
        return final_glosses

    def tokenize(self, sentence: str) -> List[str]:
        tokens = re.findall(r'\w+', sentence.lower())
        return [token for token in tokens if token not in self.stop_words]

class ImprovedStatisticalGlosser:
    def __init__(self, n=3):
        self.n = n
        self.co_occurrences = defaultdict(lambda: defaultdict(int))
        self.source_counts = defaultdict(int)
        self.target_counts = defaultdict(int)
        self.source_doc_freq = defaultdict(int)
        self.target_doc_freq = defaultdict(int)
        self.total_docs = 0
        self.stop_words: Set[str] = set()

    def train(self, source_sentences: List[str], target_sentences: List[str]):
        self.calculate_stop_words(source_sentences + target_sentences)
        self.total_docs = len(source_sentences)
        
        for source, target in zip(source_sentences, target_sentences):
            source_tokens = self.tokenize(source)
            target_tokens = self.tokenize(target)
            
            source_ngrams = self.get_ngrams(source_tokens)
            target_ngrams = self.get_ngrams(target_tokens)
            
            source_set = set(source_ngrams)
            target_set = set(target_ngrams)
            
            for s_ngram in source_ngrams:
                for t_ngram in target_ngrams:
                    self.co_occurrences[s_ngram][t_ngram] += 1
                self.source_counts[s_ngram] += 1
            
            for t_ngram in target_ngrams:
                self.target_counts[t_ngram] += 1
            
            for s_ngram in source_set:
                self.source_doc_freq[s_ngram] += 1
            for t_ngram in target_set:
                self.target_doc_freq[t_ngram] += 1

    def calculate_stop_words(self, sentences: List[str], max_stop_words: int = 100):
        word_counts = Counter(word for sentence in sentences for word in self.tokenize(sentence))
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        self.stop_words = set(word for word, _ in sorted_words[:max_stop_words])

    def tokenize(self, sentence: str) -> List[str]:
        tokens = re.findall(r'\w+', sentence.lower())
        return [token for token in tokens if token not in self.stop_words]

    def get_ngrams(self, tokens):
        return [' '.join(tokens[i:i+self.n]) for i in range(len(tokens)-self.n+1)]

    def gloss(self, source_sentence, target_sentence):
        source_tokens = self.tokenize(source_sentence)
        target_tokens = self.tokenize(target_sentence)
        
        source_ngrams = self.get_ngrams(source_tokens)
        target_ngrams = self.get_ngrams(target_tokens)
        
        mappings = []
        
        for i, s_ngram in enumerate(source_ngrams):
            ngram_mappings = []
            for j, t_ngram in enumerate(target_ngrams):
                score = self.calculate_score(s_ngram, t_ngram, i, j, len(source_ngrams), len(target_ngrams))
                if score > 0:
                    ngram_mappings.append((t_ngram, score))
            
            ngram_mappings.sort(key=lambda x: x[1], reverse=True)
            mappings.append((s_ngram, ngram_mappings[:3]))  # Keep top 3 mappings
        
        return mappings

    def calculate_score(self, source_ngram, target_ngram, source_pos, target_pos, source_len, target_len):
        epsilon = 1e-10  # Smoothing factor
        
        co_occur = self.co_occurrences[source_ngram][target_ngram] + epsilon
        source_count = self.source_counts[source_ngram] + epsilon
        target_count = self.target_counts[target_ngram] + epsilon
        
        source_idf = math.log((self.total_docs + epsilon) / (self.source_doc_freq[source_ngram] + epsilon))
        target_idf = math.log((self.total_docs + epsilon) / (self.target_doc_freq[target_ngram] + epsilon))
        
        tfidf_score = (co_occur / source_count) * source_idf * (co_occur / target_count) * target_idf
        
        position_score = 1 - abs((source_pos / source_len) - (target_pos / target_len))
        
        return tfidf_score * position_score