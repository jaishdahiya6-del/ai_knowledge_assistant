"""A local, free, retrieval-based AI assistant.
No paid API keys, no internet calls. It answers questions using the
user's own stored notes via TF-IDF + cosine similarity, plus simple
rule-based intents (stats, summaries, recommendations)."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeAssistant:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True) if not df.empty else df
        self.vectorizer = None
        self.doc_matrix = None
        if not self.df.empty:
            corpus = (self.df["title"] + " " + self.df["content"] + " " + self.df["category"]).tolist()
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=800)
            self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def _retrieve(self, query, top_k=3):
        if self.vectorizer is None:
            return pd.DataFrame()
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        top_idx = [i for i in top_idx if sims[i] > 0.0]
        if not top_idx:
            return pd.DataFrame()
        result = self.df.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result

    def answer(self, query: str) -> str:
        if self.df is None or self.df.empty:
            return "Your knowledge base is empty. Add some notes or generate sample data first."

        q = query.lower().strip()

        # Rule-based intents
        if re.search(r"how many (notes|entries)", q):
            return f"You have **{len(self.df)}** entries stored in your knowledge base."

        if "total hours" in q or ("hours" in q and "total" in q):
            total = self.df["hours_spent"].sum()
            return f"You've logged a total of **{total:.1f} hours** across all entries."

        if "categories" in q or "category" in q:
            cats = self.df["category"].value_counts()
            lines = "\n".join(f"- **{c}**: {n} entries" for c, n in cats.items())
            return f"Here's your category breakdown:\n\n{lines}"

        if "most" in q and ("time" in q or "hours" in q):
            top_cat = self.df.groupby("category")["hours_spent"].sum().idxmax()
            hrs = self.df.groupby("category")["hours_spent"].sum().max()
            return f"You've spent the most time on **{top_cat}** ({hrs:.1f} hours total)."

        if "recommend" in q or "what should i study" in q or "what should i learn" in q:
            cats = self.df["category"].value_counts()
            least = cats.idxmin()
            return (f"Based on your history, **{least}** has the fewest entries "
                    f"({cats.min()}), so it might be worth revisiting to balance your learning.")

        if "average priority" in q or ("average" in q and "priority" in q):
            avg = self.df["priority"].mean()
            return f"Your average priority rating across all entries is **{avg:.2f} / 5**."

        # Retrieval-based fallback: find most relevant notes
        matches = self._retrieve(query, top_k=3)
        if matches.empty:
            return ("I couldn't find anything closely related to that in your stored notes. "
                    "Try rephrasing, or add more notes on this topic.")

        response_lines = ["Here's what I found in your notes:\n"]
        for _, row in matches.iterrows():
            response_lines.append(
                f"**{row['title']}** (*{row['category']}*, relevance {row['similarity']:.2f})\n> {row['content']}\n"
            )
        return "\n".join(response_lines)
