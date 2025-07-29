import os
from typing import List
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from rapidfuzz import process, fuzz

class DialogModule:
    """Interactive helper to link user questions to the database schema, with three modes of matching."""

    def __init__(
        self,
        schema_elements: List[str],
        memory_path: str = "data/dialog_memory.txt",
        mode: str = 'normal'  # 'light', 'normal', or 'advanced'
    ):
        self.schema_elements = schema_elements
        self.memory_path = memory_path
        self.memory = self._load_memory()
        self.mode = mode

        # Keyword extractor and TF-IDF are always used
        self.keyword_extractor = yake.KeywordExtractor(lan='fr', top=15)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words=['french'])

        # Initialize according to mode
        if self.mode in {'normal', 'advanced'}:
            # Dense embedding model
            embedding_model = 'distiluse-base-multilingual-cased-v2'
            self.embedder = SentenceTransformer(embedding_model)
            self.schema_embeddings = self.embedder.encode(
                self.schema_elements, convert_to_tensor=True
            )
        if self.mode == 'advanced':
            # Cross-encoder model for reranking
            cross_encoder_model = 'cross-encoder/quora-roberta-base'
            self.cross_encoder = CrossEncoder(cross_encoder_model)

    def _load_memory(self) -> List[str]:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def _save_memory(self) -> None:
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            for q in self.memory:
                f.write(q + "\n")

    def add_to_memory(self, question: str) -> None:
        q = question.strip()
        if q.lower() not in {m.lower() for m in self.memory}:
            self.memory.append(q)
            self._save_memory()

    def extract_keywords(self, question: str) -> List[str]:
        raw = self.keyword_extractor.extract_keywords(question)
        return [phrase for phrase, _ in raw]

    def rank_by_tfidf(self, candidates: List[str], top_n: int = 8) -> List[str]:
        if self.memory == []:
            return candidates[:top_n]
        self.tfidf.fit(self.memory)
        analyzer = self.tfidf.build_analyzer()
        idf = self.tfidf.idf_
        vocab = self.tfidf.vocabulary_
        scored = []
        for phrase in candidates:
            tokens = analyzer(phrase)
            vals = [idf[vocab[t]] for t in tokens if t in vocab]
            avg_idf = sum(vals) / len(vals) if vals else 0.0
            scored.append((phrase, avg_idf))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in scored[:top_n]]

    def fuzzy_match_schema(self, keywords: List[str], cutoff: int = 70) -> List[dict]:
        matches = []
        for kw in keywords:
            result = process.extractOne(
                kw, self.schema_elements,
                scorer=fuzz.WRatio,
                score_cutoff=cutoff
            )
            if result:
                elem, score, _ = result
                matches.append({
                    'keyword': kw,
                    'schema_element': elem,
                    'score': score
                })
        return matches

    def semantic_match(self, phrases: List[str], top_k: int = 3, min_score: float = 0.5) -> List[dict]:
        results = []
        phrase_embeds = self.embedder.encode(phrases, convert_to_tensor=True)
        cos_scores = util.cos_sim(phrase_embeds, self.schema_embeddings)
        for i, phrase in enumerate(phrases):
            top_indices = cos_scores[i].topk(k=top_k).indices
            for idx in top_indices:
                score = float(cos_scores[i][idx])
                if score >= min_score:
                    results.append({
                        'keyword': phrase,
                        'schema_element': self.schema_elements[int(idx)],
                        'score': score * 100
                    })
        return results

    def cross_rerank(self, phrases: List[str], candidates: List[str], top_k: int = 3) -> List[dict]:
        reranked = []
        for phrase in phrases:
            inputs = [(phrase, cand) for cand in candidates]
            scores = self.cross_encoder.predict(inputs)
            top = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
            for elem, score in top:
                reranked.append({
                    'keyword': phrase,
                    'schema_element': elem,
                    'score': score * 100
                })
        return reranked

    def schema_link(self, question: str) -> List[dict]:
        self.add_to_memory(question)
        keywords = self.extract_keywords(question)
        top_phrases = self.rank_by_tfidf(keywords)

        # for 'normal' and 'advanced', do semantic retrieval
        if self.mode in {'normal', 'advanced'}:
            semantic_hits = self.semantic_match(top_phrases, top_k=10, min_score=0.0)
            if self.mode == 'normal': return semantic_hits # normal mode
            candidates = list({hit['schema_element'] for hit in semantic_hits})
            return self.cross_rerank(top_phrases, candidates) # advanced mode

        return self.fuzzy_match_schema(top_phrases) # light mode

    def ask(self, prefix: str = "Question: ", prompt: str | None = None) -> str:
        """Prompt the user and return the entered text."""
        if prompt is not None:
            print(prompt)
        return input(prefix).strip()

    def run(self) -> None:
        attempt = 1
        while True:
            question = self.ask()
            if question.lower() in {"exit", "quit"}:
                break
            links = self.schema_link(question)
            for m in links:
                print(f"{m['keyword']} -> {m['schema_element']} ({m['score']}%)")
            self.add_to_memory(question)
            attempt += 1