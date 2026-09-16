"""Exploratory Data Analysis helpers using Pandas / NumPy / Plotly."""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric = df[["priority", "hours_spent"]]
    return numeric.describe().T


def category_distribution_fig(df: pd.DataFrame):
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    fig = px.bar(counts, x="category", y="count", color="category", title="Entries by Category")
    fig.update_layout(showlegend=False, template="plotly_dark")
    return fig


def hours_over_time_fig(df: pd.DataFrame):
    d = df.copy()
    d["created_at"] = pd.to_datetime(d["created_at"])
    d["date"] = d["created_at"].dt.date
    daily = d.groupby("date")["hours_spent"].sum().reset_index()
    fig = px.line(daily, x="date", y="hours_spent", title="Hours Spent Learning Over Time", markers=True)
    fig.update_layout(template="plotly_dark")
    return fig


def priority_vs_hours_fig(df: pd.DataFrame):
    fig = px.scatter(
        df, x="priority", y="hours_spent", color="category",
        size="hours_spent", hover_data=["title"],
        title="Priority vs Hours Spent"
    )
    fig.update_layout(template="plotly_dark")
    return fig


def correlation_heatmap_fig(df: pd.DataFrame):
    numeric = df[["priority", "hours_spent"]].copy()
    numeric["title_len"] = df["title"].str.len()
    numeric["content_len"] = df["content"].str.len()
    corr = numeric.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="Viridis", zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}"
    ))
    fig.update_layout(title="Correlation Heatmap", template="plotly_dark")
    return fig


def category_avg_hours(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("category")["hours_spent"].agg(["mean", "sum", "count"]).round(2).sort_values("sum", ascending=False)
