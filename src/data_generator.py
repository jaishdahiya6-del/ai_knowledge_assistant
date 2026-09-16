"""Generates realistic sample knowledge-base entries."""
import numpy as np
from datetime import datetime, timedelta

CATEGORIES = ["Python", "Data Science", "Machine Learning", "Deep Learning", "Web Dev", "Career", "Math", "NLP"]

TITLES = {
    "Python": ["List comprehensions", "Decorators explained", "Generators vs iterators", "Virtual environments", "Async IO basics"],
    "Data Science": ["Pandas groupby tricks", "Handling missing data", "EDA checklist", "Feature scaling", "Outlier detection"],
    "Machine Learning": ["Random forest notes", "Cross-validation guide", "Bias-variance tradeoff", "Hyperparameter tuning", "Model evaluation metrics"],
    "Deep Learning": ["Backpropagation intuition", "CNN architectures", "PyTorch autograd", "Dropout regularization", "Transfer learning tips"],
    "Web Dev": ["Streamlit layout tips", "REST API design", "Flask vs FastAPI", "Frontend state management", "CSS grid basics"],
    "Career": ["Resume for data roles", "Interview prep notes", "Portfolio project ideas", "Networking tips", "Salary negotiation"],
    "Math": ["Linear algebra for ML", "Probability refresher", "Calculus for gradients", "Statistics fundamentals", "Matrix decompositions"],
    "NLP": ["Tokenization methods", "TF-IDF explained", "Word embeddings", "Sentiment analysis notes", "Named entity recognition"],
}

CONTENT_TEMPLATES = [
    "Studied {topic} today. Key insight: understanding the underlying mechanics helps a lot with debugging and design decisions.",
    "Reviewed {topic}. Practiced with a small project and it clarified several open questions I had.",
    "Deep dive into {topic}. Found it useful for improving both code quality and model performance.",
    "Quick notes on {topic}: still need more practice but the fundamentals are clearer now.",
    "Worked through examples of {topic}. This will be very useful for upcoming projects and interviews.",
    "Struggled a bit with {topic} initially, but after revisiting the docs it finally clicked.",
    "Explored {topic} in depth with hands-on coding. Confident I can apply this in real projects now.",
]


def generate_sample_data(n=120, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    start_date = datetime.now() - timedelta(days=180)
    for i in range(n):
        category = rng.choice(CATEGORIES)
        title = rng.choice(TITLES[category])
        template = rng.choice(CONTENT_TEMPLATES)
        content = template.format(topic=title.lower())
        priority = int(rng.integers(1, 6))
        hours = float(np.round(rng.uniform(0.25, 6.0), 2))
        days_offset = int(rng.integers(0, 180))
        created_at = (start_date + timedelta(days=days_offset)).isoformat(timespec="seconds")
        rows.append((f"{title} #{i+1}", category, content, priority, hours, created_at))
    return rows
