# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Keyword→Schema Linking Pipeline
# ─────────────────────────────────────────────────────────────────────────────

from numpy import average
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import process, fuzz

# Local import used only in the example section below.  When this module is
# imported as part of the library, the relative path ensures it resolves
# correctly.
from .DBManager import DBManager

def fit_tfidf(corpus, ngram_range=(1, 3), language=['french']):
    """
    1) Build & fit a TF-IDF vectorizer on your domain corpus.
    - corpus: list of strings (e.g. past user queries, docs, etc.)
    - ngram_range: consider unigrams→trigrams
    - language: for built-in stopword removal
    """
    vec = TfidfVectorizer(ngram_range=ngram_range, stop_words=language)
    vec.fit(corpus)
    return vec

def extract_candidates_yake(question, max_kw=15, language='fr'):
    """
    2) Run YAKE to get up to max_kw candidate phrases.
    Returns a list of phrases (strings).
    """
    kw_extractor = yake.KeywordExtractor(lan=language, top=max_kw)
    raw = kw_extractor.extract_keywords(question)
    # raw is [(phrase, score), ...], score=lower→better
    return [phrase for phrase, _ in raw]

def rank_candidates_by_tfidf(candidates, vectorizer, top_n=8):
    """
    3) For each candidate phrase, compute an average TF-IDF weight:
       - tokenize phrase into vectorizer’s analyzer tokens
       - look up each token’s IDF (inverse document frequency)
       - average them → phrase score
    Returns the top_n phrases sorted by descending TF-IDF.
    """
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

def fuzzy_match_schema(keywords, schema_elements, cutoff=70):
    """
    4) For each keyword, find the best fuzzy match in schema_elements:
       - uses WRatio (rapidfuzz’s weighted Levenshtein)
       - discards matches below cutoff%
    Returns a list of dicts: { keyword, matched_element, score }.
    """
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

def schema_link(question, corpus, schema_elements):
    """
    Full pipeline:
      • Fit TF-IDF on corpus
      • Extract YAKE candidates from question
      • Re-rank by TF-IDF
      • Fuzzy-match top phrases to schema
    Returns list of matches.
    """
    tfidf = fit_tfidf(corpus)
    candidates = extract_candidates_yake(question)
    top_phrases = rank_candidates_by_tfidf(candidates, tfidf)
    links = fuzzy_match_schema(top_phrases, schema_elements)
    return links

# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────

TRESHOLD = 70  # minimum score for a match to be considered relevant

def main():
        # 1. your past queries / documentation
    corpus = [
        "montre-moi le salaire moyen par département",
        "liste des employés embauchés après 2015",
        "nombre total de projets par manager",
        # … etc.
    ]

    db = DBManager("data/sqlite/employee_db.sqlite")

    # schema = [ # list of tables or columns in your database
    #     "employees", "departments", "projects", 
    #     "salary", "hire_date", "manager_id", 
    #     "project_count", "department_name",
    # ]

    schema = db.extract_column_table_pairs()

    question = "Quel est le salaire moyen des employés par département ?"
    #question = "Quel est la cité de chaque département ?"

    schema_elements = list(schema.keys())
    for table, cols in schema.items():
        for col in cols:
            schema_elements.append(f"{col} {table}")

    print(f"Schema elements: {schema_elements}")

    matches = schema_link(question, corpus, schema_elements)

    for m in matches:
        meta = m["schema_element"].split(" ")
        if len(meta) > 1:
            m["schema_table"] = meta[1]
            m["schema_column"] = meta[0]
        else:
            m["schema_table"] = meta[0]

    sorted_matches = sorted(matches, key=lambda x: x['score'], reverse=True)
    selected_entries = []
    selected_tables = []
    for m in sorted_matches:
        if m['schema_table'] not in selected_tables and m['score'] > TRESHOLD:
            selected_entries.append(m)
            selected_tables.append(m['schema_table'])
        print(f"Matched: '{m['keyword']}' → '{m['schema_table']}.{m.get('schema_column',"")}' ({m['score']}%)")

    print(f"Found {len(selected_entries)} relevant schema links : {', '.join([table for table in selected_tables])}")    
    average_score = average([m['score'] for m in selected_entries])
    selected_tables = [m['schema_element'] for m in selected_entries if m['schema_element'] in schema]
    return average_score, selected_tables

if __name__ == "__main__":
    main()
    
    
