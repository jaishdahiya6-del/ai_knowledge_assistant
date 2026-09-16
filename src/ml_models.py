"""Classical ML: predict priority level from entry features using scikit-learn."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def build_features(df: pd.DataFrame):
    d = df.copy()
    d["title_len"] = d["title"].str.len()
    d["content_len"] = d["content"].str.len()
    d["word_count"] = d["content"].str.split().apply(len)
    le = LabelEncoder()
    d["category_enc"] = le.fit_transform(d["category"])
    return d, le


def train_priority_classifier(df: pd.DataFrame, min_rows=20):
    """Predicts 'priority' bucket (Low/Med/High) from category + text length + hours."""
    if len(df) < min_rows:
        return None

    d, le = build_features(df)
    d["priority_bucket"] = pd.cut(d["priority"], bins=[0, 2, 3, 5], labels=["Low", "Medium", "High"])

    X = d[["category_enc", "title_len", "content_len", "word_count", "hours_spent"]]
    y = d["priority_bucket"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    return {
        "model": clf,
        "scaler": scaler,
        "label_encoder": le,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "classes": clf.classes_,
        "feature_importances": importances,
    }


def train_hours_regressor(df: pd.DataFrame, min_rows=20):
    """Predicts hours_spent from category + text features via Linear Regression."""
    if len(df) < min_rows:
        return None

    d, le = build_features(df)
    X = d[["category_enc", "title_len", "content_len", "word_count", "priority"]]
    y = d["hours_spent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": reg,
        "label_encoder": le,
        "mae": mae,
        "r2": r2,
        "y_test": y_test.values,
        "y_pred": y_pred,
    }


def predict_priority(bundle, category, title, content, hours_spent, le_category):
    title_len = len(title)
    content_len = len(content)
    word_count = len(content.split())
    try:
        category_enc = le_category.transform([category])[0]
    except ValueError:
        category_enc = 0
    X = np.array([[category_enc, title_len, content_len, word_count, hours_spent]])
    X_scaled = bundle["scaler"].transform(X)
    pred = bundle["model"].predict(X_scaled)[0]
    proba = bundle["model"].predict_proba(X_scaled)[0]
    return pred, dict(zip(bundle["model"].classes_, proba))
