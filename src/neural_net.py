"""A small feed-forward neural network (PyTorch) that predicts priority
from numeric + categorical features. Trains fully in-memory, no GPU required."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

torch.manual_seed(42)


class PriorityNet(nn.Module):
    def __init__(self, input_dim, hidden1=32, hidden2=16, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def prepare_tensors(df: pd.DataFrame):
    d = df.copy()
    d["title_len"] = d["title"].str.len()
    d["content_len"] = d["content"].str.len()
    d["word_count"] = d["content"].str.split().apply(len)

    le_cat = LabelEncoder()
    d["category_enc"] = le_cat.fit_transform(d["category"])

    d["priority_bucket"] = pd.cut(d["priority"], bins=[0, 2, 3, 5], labels=[0, 1, 2]).astype(int)

    X = d[["category_enc", "title_len", "content_len", "word_count", "hours_spent"]].values.astype(np.float32)
    y = d["priority_bucket"].values.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    return X_scaled, y, scaler, le_cat


def train_neural_net(df: pd.DataFrame, epochs=60, lr=0.01, min_rows=20):
    if len(df) < min_rows:
        return None

    X, y, scaler, le_cat = prepare_tensors(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    model = PriorityNet(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    test_losses = []
    test_accuracies = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t).item()
            test_losses.append(test_loss)
            preds = torch.argmax(test_outputs, dim=1)
            acc = (preds == y_test_t).float().mean().item()
            test_accuracies.append(acc)
        model.train()

    model.eval()
    with torch.no_grad():
        final_preds = torch.argmax(model(X_test_t), dim=1).numpy()

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le_cat,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
        "final_accuracy": test_accuracies[-1],
        "y_test": y_test,
        "y_pred": final_preds,
    }


def predict_with_net(bundle, category, title, content, hours_spent):
    title_len = len(title)
    content_len = len(content)
    word_count = len(content.split())
    try:
        category_enc = bundle["label_encoder"].transform([category])[0]
    except ValueError:
        category_enc = 0

    X = np.array([[category_enc, title_len, content_len, word_count, hours_spent]], dtype=np.float32)
    X_scaled = bundle["scaler"].transform(X).astype(np.float32)
    X_t = torch.tensor(X_scaled)

    bundle["model"].eval()
    with torch.no_grad():
        logits = bundle["model"](X_t)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_class = int(np.argmax(probs))

    labels = {0: "Low", 1: "Medium", 2: "High"}
    return labels[pred_class], {labels[i]: float(p) for i, p in enumerate(probs)}
