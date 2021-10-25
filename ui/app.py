import sys, os, pathlib
# Ensure project root is on PYTHONPATH for 'src' package imports
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

from src.nlp_processor import NLPProcessor

DATA_DIR = Path("data/processed")
INSIGHTS_FILE = DATA_DIR / "mda_insights.jsonl"
API_BASE = "http://127.0.0.1:8000"

COMPANY_NAME_MAP = {
    "0000320193": "Apple Inc.",
    "0000070858": "Procter & Gamble",
    "0000796343": "PepsiCo, Inc.",
}

st.set_page_config(page_title="DeepFinNER Dashboard", layout="wide")
st.title("DeepFinNER – MD&A Insight Explorer")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_insight_index() -> pd.DataFrame:
    recs: List[Dict] = []
    if INSIGHTS_FILE.exists():
        with INSIGHTS_FILE.open("r", encoding="utf-8") as fp:
            for line in fp:
                obj = json.loads(line)
                cik = obj["company_cik"]
                recs.append({
                    "cik": cik,
                    "name": COMPANY_NAME_MAP.get(cik, cik),
                    "year": obj["filing_date"][:4]
                })
    df = pd.DataFrame(recs).drop_duplicates().sort_values(["name", "year"])
    return df


def fetch_insight(cik: str, year: str) -> Dict:
    url = f"{API_BASE}/insights/{cik}?year={year}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def fetch_graph(cik: str, year: str) -> Dict:
    url = f"{API_BASE}/graph/{cik}?year={year}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def _edge_color(data):
    pos, neg = data.get("sent_positive", 0), data.get("sent_negative", 0)
    if pos > neg:
        return "#2ca02c"  # green
    if neg > pos:
        return "#d62728"  # red
    return "#888888"  # neutral


def graph_to_plotly(graph_json: Dict) -> go.Figure:
    G = nx.node_link_graph(graph_json)
    pos = nx.spring_layout(G, k=0.8, seed=42)

    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                line=dict(width=1, color=_edge_color(data)),
                hoverinfo="text",
                text=[f"{u} → {v}<br>verbs: {', '.join(data['verbs'])}"],
                mode="lines",
                showlegend=False,
            )
        )

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=10, color="#1f78b4"))

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(showlegend=False,
                                     margin=dict(l=20, r=20, t=20, b=20),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

idx_df = load_insight_index()
if idx_df.empty:
    st.error("No insight data found. Please run the pipelines first.")
    st.stop()

company_names = idx_df[["name", "cik"]].drop_duplicates().sort_values("name")
select_name = st.sidebar.selectbox("Company", company_names["name"].tolist())
select_cik = company_names[company_names["name"] == select_name]["cik"].values[0]

years = idx_df[idx_df["cik"] == select_cik]["year"].unique().tolist()
select_year = st.sidebar.selectbox("Year", years)

if st.sidebar.button("Load Insights"):
    with st.spinner("Fetching data from API …"):
        try:
            insight = fetch_insight(select_cik, select_year)
            graph_json = fetch_graph(select_cik, select_year)
        except Exception as exc:
            st.error(f"API error: {exc}")
            st.stop()

    st.subheader("Aggregate Sentiment")
    # Executive Summary
    if "summary" in insight:
        st.subheader("Executive Summary (auto-generated)")
        st.code(insight["summary"], language="markdown")
        st.download_button(
            label="Copy summary as TXT",
            data=insight["summary"],
            file_name=f"{select_cik}_{select_year}_summary.txt",
        )

    # Aggregate sentiment metrics
    agg = insight["sentiment"]["aggregate"]
    st.metric("Positive", f"{agg['positive']*100:.1f}%")
    st.metric("Negative", f"{agg['negative']*100:.1f}%")
    st.metric("Neutral", f"{agg['neutral']*100:.1f}%")

    st.subheader("Relationship Graph (edges colored by sentiment)")
    fig = graph_to_plotly(graph_json)
    st.plotly_chart(fig, use_container_width=True)

    # Sentence sentiment table
    st.subheader("Sentence-level Sentiment")
    nlp_proc = NLPProcessor()
    sentences = nlp_proc.tokenize_text(insight["mda"], method="sentence")
    sent_labels = [s["label"] for s in insight["sentiment"]["sentence_breakdown"]]
    min_len = min(len(sentences), len(sent_labels))
    df_sent = pd.DataFrame({
        "sentence": sentences[:min_len],
        "sentiment": sent_labels[:min_len],
    })
    st.dataframe(df_sent, use_container_width=True)

    st.subheader("Entities & Metrics (raw)")
    st.json(insight["entities"]) 