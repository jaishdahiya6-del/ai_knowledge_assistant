"""NLP utilities: tokenization, keyword extraction (TF-IDF), and a
lightweight lexicon-based sentiment analyzer. Fully self-contained —
no internet downloads (no nltk.download needed) so it works offline."""
import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

STOPWORDS = set("""
a an the and or but if while is are was were be been being to of in on
for with as by at from this that these those it its it's i you he she
we they them his her our your their not no do does did doing have has
had having will would shall should can could may might must about above
after again against all am because been before below between both down
during each few further here how into more most other over own same so
some such than too very what when where which who whom why
""".split())

POSITIVE_WORDS = set("""
good great excellent clear useful helpful confident clarified success
successful improved improve improving efficient effective solid strong
enjoy enjoyed love loved insightful smooth breakthrough progress
""".split())

NEGATIVE_WORDS = set("""
bad hard difficult struggle struggled struggling confusing confused
frustrating frustrated fail failed failing weak slow problem problems
issue issues stuck unclear tough boring
""".split())


def tokenize(text: str):
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def word_frequency(df: pd.DataFrame, top_n=20):
    all_tokens = []
    for content in df["content"].fillna(""):
        all_tokens.extend(tokenize(content))
    counts = Counter(all_tokens)
    return pd.DataFrame(counts.most_common(top_n), columns=["word", "count"])


def extract_keywords_tfidf(df: pd.DataFrame, top_n=15):
    if df.empty:
        return pd.DataFrame(columns=["keyword", "score"])
    docs = df["content"].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf_matrix = vectorizer.fit_transform(docs)
    scores = tfidf_matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    result = pd.DataFrame({"keyword": terms, "score": scores}).sort_values("score", ascending=False).head(top_n)
    return result.reset_index(drop=True)


def sentiment_score(text: str):
    tokens = tokenize(text)
    if not tokens:
        return 0.0, "Neutral"
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    score = (pos - neg) / max(len(tokens), 1)
    if score > 0.02:
        label = "Positive"
    elif score < -0.02:
        label = "Negative"
    else:
        label = "Neutral"
    return round(score, 4), label


def sentiment_for_df(df: pd.DataFrame):
    d = df.copy()
    results = d["content"].fillna("").apply(sentiment_score)
    d["sentiment_score"] = results.apply(lambda x: x[0])
    d["sentiment_label"] = results.apply(lambda x: x[1])
    return d
