import os
from typing import List
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import process, fuzz

class DialogModule:
    """Interactive helper to link user questions to the database schema."""

    def __init__(self, schema_elements: List[str], memory_path: str = "data/dialog_memory.txt"):
        self.schema_elements = schema_elements
        self.memory_path = memory_path
        self.memory = self._load_memory()

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
        self.memory.append(question)
        self._save_memory()

    def fit_tfidf(self, ngram_range=(1, 3), language=['french']):
        vec = TfidfVectorizer(ngram_range=ngram_range, stop_words=language)
        vec.fit(self.memory)
        return vec

    def extract_candidates_yake(self, question, max_kw=15, language='fr'):
        kw_extractor = yake.KeywordExtractor(lan=language, top=max_kw)
        raw = kw_extractor.extract_keywords(question)
        # raw is [(phrase, score), ...], score=lower→better
        return [phrase for phrase, _ in raw]

    def rank_candidates_by_tfidf(self, candidates, vectorizer, top_n=8):
        scored = []
        analyzer = vectorizer.build_analyzer()
        idf = vectorizer.idf_
        vocab = vectorizer.vocabulary_
        for phrase in candidates:
            tokens = analyzer(phrase)
            # collect IDF for tokens that exist in vocab
            vals = [idf[vocab[t]] for t in tokens if t in vocab]
            avg_idf = sum(vals) / len(vals) if vals else 0.0
            scored.append((phrase, avg_idf))
        # sort by score desc
        scored.sort(key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in scored[:top_n]]

    def fuzzy_match_schema(self, keywords, schema_elements, cutoff=70):
        matches = []
        for kw in keywords:
            result = process.extractOne(
                kw, schema_elements,
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

    def schema_link(self, question: str):
        tfidf = self.fit_tfidf()
        candidates = self.extract_candidates_yake(question)
        top_phrases = self.rank_candidates_by_tfidf(candidates, tfidf)
        links = self.fuzzy_match_schema(top_phrases, self.schema_elements)
        return links

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

if __name__ == "__main__":
    example_schema = ["employees", "departments"]
    module = DialogModule(example_schema, "data/dialog_memory.txt")
    module.add_to_memory("montre-moi le salaire moyen par département")
    module.add_to_memory("liste des employés embauchés après 2015")
    module.add_to_memory("nombre total de projets par manager")
