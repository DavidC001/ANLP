#!/usr/bin/env python
# coding: utf-8
"""
Semantic‑role knowledge‑graph dashboard.
Entity‑matching logic reworked.
    • Normalised string key (lower, punctuation‑stripped, lemma) used as primary index.
    • Fallback cosine similarity (≥ 0.85) on BERT span embeddings to merge paraphrastic mentions.
    • Global canonical map ensures single identifier per conceptual entity across sentences.
No extra heavyweight dependencies added – uses NLTK WordNet for lemmatisation.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

sys.path.append(".")

import networkx as nx
import numpy as np
import torch
import wikipedia
from dash import Dash, Input, Output, State, dcc, html
import dash_cytoscape as cyto
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from dataloaders.NomBank_dataloader import roles as NOM_ROLES
from dataloaders.UP_dataloader import roles as UP_ROLES
from model import SRL_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REL_THRESHOLD = 0.5
ROLE_THRESHOLD = 0.4
SIM_THRESHOLD = 0.85  # relaxed for semantic duplicates

###############################################################################
# Text utilities
###############################################################################

def escape_text(t: str) -> str:
    return re.sub(r"(['\"\\])", r"\\\\\1", t)

###############################################################################
# Normalisation helpers for entity matching
###############################################################################

_punct_tbl = str.maketrans("", "", string.punctuation)
_lemma = WordNetLemmatizer()

@lru_cache(maxsize=8192)
def _canonical(text: str) -> str:
    """Lowercase, strip punctuation, lemmatise tokens, collapse whitespace."""
    toks = [
        _lemma.lemmatize(tok.lower())
        for tok in word_tokenize(text.translate(_punct_tbl))
        if tok.strip()
    ]
    return " ".join(toks)

###############################################################################
# Dense retriever
###############################################################################

class LocalRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self._embedder = SentenceTransformer(model_name)
        self._vecs: np.ndarray | None = None
        self._meta: list[dict[str, Any]] = []

    def add(self, texts: Sequence[str], metas: Sequence[dict[str, Any]]):
        vecs = self._embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self._vecs = vecs if self._vecs is None else np.vstack((self._vecs, vecs))
        self._meta.extend(metas)

    def query(self, text: str, k: int = 5):
        if self._vecs is None:
            return []
        v = self._embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = np.dot(self._vecs, v)
        if sims.size == 0:
            return []
        idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [{**self._meta[i], "score": float(sims[i])} for i in idx]

###############################################################################
# Knowledge graph
###############################################################################

class KnowledgeGraph:
    def __init__(self, retriever: LocalRetriever):
        self.g = nx.DiGraph()
        self._ret = retriever

    def add_node(self, node_id: str, text: str, role: str):
        if node_id in self.g:
            return
        self.g.add_node(node_id, label=text, role=role)
        self._ret.add([text], [{"node_id": node_id, "role": role, "text": text}])

    def add_edge(self, src: str, tgt: str, label: str):
        if not self.g.has_edge(src, tgt):
            self.g.add_edge(src, tgt, label=label)

    def elements(self, nodes: set[str] | None = None):
        g_view = self.g if nodes is None else self.g.subgraph(nodes).copy()
        el: list[dict[str, Any]] = []
        for n, d in g_view.nodes(data=True):
            el.append({"data": {"id": n, "label": d.get("label", ""), "role": d.get("role", "")}})
        for i, (u, v, d) in enumerate(g_view.edges(data=True)):
            el.append({"data": {"id": f"e_{i}", "source": u, "target": v, "label": d.get("label", "")}})
        return el

###############################################################################
# SRL helpers – improved entity matching
###############################################################################

_ent_tok = AutoTokenizer.from_pretrained("bert-large-cased")
_ent_model = AutoModel.from_pretrained("bert-large-cased")

# canonical text -> node_id
_canon_map: dict[str, str] = {}
# node id -> embedding vector (to compare if canonicalisation misses paraphrases)
_node_vecs: dict[str, torch.Tensor] = {}
_argm_map: dict[str, str] = {}


def _dedup(text: str, span_text: str, span_idx: Sequence[int], role: str, node_id: str):
    """Return existing node_id if *span_text* matches earlier entity, else given *node_id*.
    Matching tiered:
        1. Canonicalised string identity.
        2. Cosine similarity ≥ SIM_THRESHOLD between span embeddings.
    """
    if role.startswith("ARGM"):
        if span_text in _argm_map:
            return _argm_map[span_text]
        _argm_map[span_text] = node_id
        return node_id

    canon = _canonical(span_text)
    if canon in _canon_map:
        return _canon_map[canon]

    # fallback semantic similarity
    with torch.inference_mode():
        toks = _ent_tok(text, return_tensors="pt")
        rep = _ent_model(**toks).last_hidden_state[0]
        wids = toks.word_ids()
        span_tok_idx = [i for i, wid in enumerate(wids) if wid in span_idx]
        vec = rep[span_tok_idx].mean(0)
        for other_id, other_vec in _node_vecs.items():
            if torch.cosine_similarity(vec, other_vec, dim=0) >= SIM_THRESHOLD:
                _canon_map[canon] = other_id  # link canonical form for future hits
                return other_id
        # no match – register new
        _canon_map[canon] = node_id
        _node_vecs[node_id] = vec
        return node_id
    
###############################################################################
# Span extraction
###############################################################################

def _extract(tokens: Sequence[str], role_logits: torch.Tensor, roles: Sequence[str]):
    role_map: dict[int, list[int]] = {}
    for idx, logits in enumerate(role_logits):
        pos = (torch.sigmoid(logits) > ROLE_THRESHOLD).nonzero(as_tuple=True)[0]
        for r in pos.tolist():
            role_map.setdefault(r, []).append(idx)

    for r_idx, idxs in role_map.items():
        idxs.sort()
        groups: list[list[int]] = []
        for i in idxs:
            if groups and i == groups[-1][-1] + 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        for g in groups:
            span_tokens = [tokens[i] for i in g]
            yield span_tokens, g, roles[r_idx + 2]

###############################################################################
# Graph population
###############################################################################

def _populate(sentence: str, rel_logits: torch.Tensor, role_logits_batch: list[torch.Tensor], roles: Sequence[str], kg: KnowledgeGraph, sid: int):
    tok = TreebankWordTokenizer()
    toks = tok.tokenize(sentence)
    rel_idx = (torch.sigmoid(rel_logits) > REL_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
    for rel_pos, role_logits in zip(rel_idx, role_logits_batch):
        rel_id = f"rel_{sid}_{rel_pos}"
        kg.add_node(rel_id, escape_text(toks[rel_pos]), "relation")
        for span_toks, span_idx, role in _extract(toks, role_logits, roles):
            span_text = " ".join(span_toks)
            node_id = f"span_{sid}_{span_idx[0]}_{span_idx[-1]}"
            node_id = _dedup(sentence, span_text, span_idx, role, node_id)
            kg.add_node(node_id, escape_text(span_text), role)
            kg.add_edge(rel_id, node_id, role.replace("-", "_"))

###############################################################################
# QA
###############################################################################

def answer_with_ctx(q: str, kg: KnowledgeGraph, lm: str = "meta-llama/Llama-3.2-1B", k: int = 5):
    ctx = kg._ret.query(q, k)
    if not ctx:
        return "No answer found.", set()
    ctx_nodes = {c["node_id"] for c in ctx}
    prompt_ctx = "\n".join(f"- {c['text']}" for c in ctx)
    prompt = f"Context:\n{prompt_ctx}\n\nQuestion: {q}\nAnswer:"
    tok = AutoTokenizer.from_pretrained(lm)
    model = AutoModelForCausalLM.from_pretrained(lm, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=256, do_sample=False)
    ans = tok.decode(out[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    return ans, ctx_nodes

###############################################################################
# Builder
###############################################################################

def build(source: str, *, roles: Sequence[str], srl: SRL_MODEL, kg: KnowledgeGraph):
    if source.lower() == "w":
        title = input("Wikipedia title: ")
        text = wikipedia.page(title).content
    else:
        text = Path(source).read_text(encoding="utf-8")

    for i, sent in enumerate(sent_tokenize(text)):
        rel_logits, _, role_logits = srl.inference(sent)
        if role_logits:
            _populate(sent, rel_logits, role_logits[0], roles, kg, i)

    logging.info("Nodes: %d | Edges: %d", kg.g.number_of_nodes(), kg.g.number_of_edges())

###############################################################################
# Dash UI
###############################################################################

def serve(kg: KnowledgeGraph):
    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Input(id="q", type="text", placeholder="Ask"),
        html.Button("Run", id="btn"),
        html.Pre(id="ans"),
        cyto.Cytoscape(id="graph", elements=kg.elements(), layout={"name": "cose"}, style={"width": "100%", "height": "800px"}, stylesheet=[{"selector": "node", "style": {"label": "data(label)", "background-color": "skyblue"}}, {"selector": "[role = 'relation']", "style": {"background-color": "red"}}, {"selector": "edge", "style": {"label": "data(label)", "text-rotation": "autorotate", "text-margin-y": "-10px"}}]),
    ])

    @app.callback(
        Output("ans", "children"),
        Output("graph", "elements"),
        Input("btn", "n_clicks"),
        State("q", "value"),
        prevent_initial_call=True,
    )
    def _qa(_, q):  # noqa: ANN001
        if not q:
            return "", kg.elements()
        ans, ctx_nodes = answer_with_ctx(q, kg)
        # Expand subgraph to include one‑hop neighbours of context nodes for clarity
        expanded: set[str] = set(ctx_nodes)
        for n in ctx_nodes:
            expanded.update(nx.all_neighbors(kg.g, n))
        return ans, kg.elements(expanded)

    app.run()

###############################################################################
# Main
###############################################################################

def main():
    p = argparse.ArgumentParser("SRL‑KG Dashboard")
    p.add_argument("model")
    p.add_argument("roles", choices=["UP", "NOM"])
    p.add_argument("source")
    p.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.getLogger().setLevel(args.log)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads(Path(f"models/{args.model}.json").read_text())
    cfg["device"] = device
    srl = SRL_MODEL(**cfg)
    srl.load_state_dict(torch.load(f"models/{args.model}.pt", map_location=device))
    srl.eval()

    roles = UP_ROLES if args.roles == "UP" else NOM_ROLES

    kg = KnowledgeGraph(LocalRetriever())
    build(args.source, roles=roles, srl=srl, kg=kg)
    serve(kg)

if __name__ == "__main__":
    main()
