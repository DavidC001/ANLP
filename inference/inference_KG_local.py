from __future__ import annotations
import argparse, json, logging, re, string, sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

sys.path.append(".")

# ───────────────────────── third‑party
import networkx as nx, numpy as np, torch, wikipedia
from dash import Dash, Input, Output, State, dcc, html
import dash_cytoscape as cyto
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# ───────────────────────── project‑local
from dataloaders.NomBank_dataloader import roles as NOM_ROLES
from dataloaders.UP_dataloader import roles as UP_ROLES
from model import SRL_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REL_THRESHOLD, ROLE_THRESHOLD, SIM_THRESHOLD = 0.5, 0.4, 0.85
DISPLAY_LIMIT = 40                            # visible chars in node label


# ════════════════════════ helpers ════════════════════════════════════════
def escape_text(t: str) -> str:
    return re.sub(r"(['\"\\])", r"\\\\\1", t)


_punct_tbl = str.maketrans("", "", string.punctuation)
_lemma = WordNetLemmatizer()

@lru_cache(maxsize=65536)
def _canonical(text: str) -> str:
    toks = (_lemma.lemmatize(tok.lower())
            for tok in word_tokenize(text.translate(_punct_tbl))
            if tok.strip())
    return " ".join(toks)


# ════════════════════════ Hybrid retriever ═══════════════════════════════
class LocalRetriever:
    """α·cosine + (1‑α)·Jaccard, optional role filtering."""
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 alpha: float = 0.7):
        self._embedder = SentenceTransformer(model_name)
        self._alpha = alpha
        self._vecs: np.ndarray | None = None
        self._meta: list[dict[str, Any]] = []
        self._lex:  list[set[str]]        = []

    def add(self, texts: Sequence[str], metas: Sequence[dict[str, Any]]):
        vecs = self._embedder.encode(texts, convert_to_numpy=True,
                                     normalize_embeddings=True)
        self._vecs = vecs if self._vecs is None else np.vstack((self._vecs, vecs))
        self._meta.extend(metas)
        self._lex.extend([{w.lower() for w in word_tokenize(t)} for t in texts])

    def query(self, text: str, k: int = 5, role_filter: str | None = None):
        if self._vecs is None:
            return []
        v   = self._embedder.encode([text], convert_to_numpy=True,
                                    normalize_embeddings=True)[0]
        cos = np.dot(self._vecs, v)

        q_lex = {w.lower() for w in word_tokenize(text)}
        jac   = np.array([len(q_lex & l) / len(q_lex | l) if q_lex | l else 0.0
                          for l in self._lex], dtype=np.float32)

        score = self._alpha * cos + (1 - self._alpha) * jac
        order = np.argsort(-score)

        hits: list[dict[str, Any]] = []
        for i in order:
            if role_filter is None or self._meta[i]["role"] == role_filter:
                hits.append({**self._meta[i], "score": float(score[i])})
                if len(hits) == k:
                    break
        return hits


# ════════════════════════ Knowledge graph ════════════════════════════════
class KnowledgeGraph:
    def __init__(self, retriever: LocalRetriever):
        self.g  = nx.DiGraph()
        self._r = retriever

    def add_node(self, node_id: str, label: str, role: str, full: str):
        if node_id in self.g: return
        self.g.add_node(node_id, label=label, full=full, role=role)
        self._r.add([full], [{"node_id": node_id, "role": role, "text": full}])

    def add_edge(self, src: str, tgt: str, label: str):
        if not self.g.has_edge(src, tgt):
            self.g.add_edge(src, tgt, label=label)

    def elements(self, nodes: set[str] | None = None):
        g = self.g if nodes is None else self.g.subgraph(nodes).copy()
        els: list[dict[str, Any]] = []
        for n, d in g.nodes(data=True):
            els.append({"data": {"id": n,
                                 "label": d["label"],
                                 "full":  d["full"],
                                 "role":  d["role"]}})
        for i, (u, v, d) in enumerate(g.edges(data=True)):
            els.append({"data": {"id": f"e_{i}", "source": u, "target": v,
                                 "label": d["label"]}})
        return els


# ═════════════════ entity de‑duplication ═════════════════════════════════
_ent_tok   = AutoTokenizer.from_pretrained("bert-large-cased")
_ent_model = AutoModel.from_pretrained("bert-large-cased")

_canon_map: dict[str, str]         = {}
_node_vecs: dict[str, torch.Tensor] = {}
_argm_cache: dict[str, str]         = {}


def _dedup(sent: str, span: str, idx: Sequence[int],
           role: str, node_id: str):
    if role.startswith("ARGM"):
        return _argm_cache.setdefault(span, node_id)

    canon = _canonical(span)
    if canon in _canon_map:
        return _canon_map[canon]

    with torch.inference_mode():
        toks = _ent_tok(sent, return_tensors="pt")
        rep  = _ent_model(**toks).last_hidden_state[0]
        wids = toks.word_ids()
        span_tok = [i for i, wid in enumerate(wids) if wid in idx]
        vec      = rep[span_tok].mean(0)
        for oid, ovec in _node_vecs.items():
            if torch.cosine_similarity(vec, ovec, 0) >= SIM_THRESHOLD:
                _canon_map[canon] = oid
                return oid
        _canon_map[canon], _node_vecs[node_id] = node_id, vec
        return node_id


# ═════════════════════ span extractor ════════════════════════════════════
def _extract(tokens: Sequence[str], role_logits: torch.Tensor,
             roles: Sequence[str]):
    probs = torch.sigmoid(role_logits)              # [T,R]
    best  = [-1] * len(tokens)
    for i, p in enumerate(probs):
        v, ridx = torch.max(p, 0)
        if v > ROLE_THRESHOLD:
            best[i] = int(ridx)

    cur_role: int | None = None
    cur_idx: list[int]   = []

    def flush():
        nonlocal cur_role, cur_idx
        if cur_role is None: return
        span_prob = probs[cur_idx, cur_role].mean().item()
        if span_prob >= ROLE_THRESHOLD:
            yield [tokens[j] for j in cur_idx], cur_idx, roles[cur_role + 2]
        cur_role, cur_idx = None, []

    for i, r in enumerate(best):
        if r == cur_role:
            cur_idx.append(i)
        else:
            yield from flush()
            if r != -1:
                cur_role, cur_idx = r, [i]
    yield from flush()


# ═══════════════════ graph population ════════════════════════════════════
def _populate(sent: str, rel_logits: torch.Tensor,
              role_batches: list[torch.Tensor], roles: Sequence[str],
              kg: KnowledgeGraph, sid: int):
    tok    = TreebankWordTokenizer()
    tokens = tok.tokenize(sent)
    rel_idcs = (torch.sigmoid(rel_logits) > REL_THRESHOLD).nonzero(
                   as_tuple=True)[0].tolist()

    for rel_pos, role_log in zip(rel_idcs, role_batches):
        rel_id   = f"rel_{sid}_{rel_pos}"
        rel_full = escape_text(tokens[rel_pos])
        rel_lbl  = rel_full[:DISPLAY_LIMIT] + ("…" if len(rel_full) > DISPLAY_LIMIT else "")
        kg.add_node(rel_id, rel_lbl, "relation", rel_full)

        span_recs: list[tuple[str, str]] = []
        for span_tok, span_idx, role in _extract(tokens, role_log, roles):
            span_txt = " ".join(span_tok)
            node_id  = f"span_{sid}_{span_idx[0]}_{span_idx[-1]}"
            node_id  = _dedup(sent, span_txt, span_idx, role, node_id)

            span_lbl = span_txt[:DISPLAY_LIMIT] + ("…" if len(span_txt) > DISPLAY_LIMIT else "")
            kg.add_node(node_id, span_lbl, role, escape_text(span_txt))
            kg.add_edge(rel_id, node_id, role)
            span_recs.append((role, span_txt))

        # drop relations that never got an argument
        if kg.g.out_degree(rel_id) == 0:
            kg.g.remove_node(rel_id)
            continue

        cluster_txt = " ; ".join([tokens[rel_pos]] +
                                 [f"{r} {t}" for r, t in span_recs])
        kg._r.add([cluster_txt],
                  [{"node_id": rel_id, "role": "relation", "text": cluster_txt}])


# ═════════════════════ QA utilities ══════════════════════════════════════
def _rel_cluster(node: str, kg: KnowledgeGraph) -> tuple[str, set[str]]:
    """Return (root relation, whole cluster)."""
    if kg.g.nodes[node]["role"] == "relation":
        root = node
    else:
        preds = list(kg.g.predecessors(node))
        root  = preds[0] if preds and kg.g.nodes[preds[0]]["role"] == "relation" else None
    return (root or node,
            {root, *kg.g.successors(root)} if root else {node})


def _names_in_question(q: str, kg: KnowledgeGraph) -> list[str]:
    q_tokens = {w.lower() for w in word_tokenize(q)}
    return [n for n, d in kg.g.nodes(data=True)
            if d["role"] != "relation" and
               any(tok.lower() in q_tokens for tok in d["full"].split())]

qa_tok = None
qa_model = None
def answer_with_ctx(q: str, kg: KnowledgeGraph,
                    lm: str = "meta-llama/Llama-3.2-1B-Instruct", k: int = 5):
    global qa_tok, qa_model
    # graph‑based shortcut if ≥ 2 entity names mentioned
    ents = _names_in_question(q, kg)
    if len(ents) >= 2:
        rels: set[str] = set()
        for a in ents:
            for b in ents:
                if a == b: continue
                if kg.g.has_edge(a, b): rels.add(a)
                if kg.g.has_edge(b, a): rels.add(b)
                preds = set(kg.g.predecessors(a)) & set(kg.g.predecessors(b))
                rels |= {p for p in preds if kg.g.nodes[p]["role"] == "relation"}
        hits = [{"node_id": r, "role": "relation", "text": kg.g.nodes[r]["full"]}
                for r in rels]
    else:
        # dense retriever – prefer relation clusters first
        hits = kg._r.query(q, k=k, role_filter="relation") \
            or kg._r.query(q, k=k)

    if not hits:
        return "No answer found.", set()

    clusters: dict[str, set[str]] = {}
    for h in hits:
        root, grp = _rel_cluster(h["node_id"], kg)
        clusters.setdefault(root, set()).update(grp)
    ctx_nodes = set().union(*clusters.values())

    ctx_lines: list[str] = []
    for root, grp in clusters.items():
        pred = kg.g.nodes[root]["full"]
        args = (f"{kg.g.edges[root, tgt]['label']}={kg.g.nodes[tgt]['full']}"
                for tgt in grp if tgt != root)
        ctx_lines.append(f"{pred}: " + "; ".join(sorted(args)))
    prompt_ctx = "\n".join(ctx_lines)

    prompt = f"Context:\n{prompt_ctx}\n\nQuestion: {q}"
    chat = [
        {"role": "system", "content": "You are a helpful assistant. Use the context to answer the question. If the context is not enough, say 'I don't know'."},
        {"role": "user", "content": prompt},
    ]
    
    if qa_tok is None or qa_model is None:
        tok    = AutoTokenizer.from_pretrained(lm)
        model  = AutoModelForCausalLM.from_pretrained(
                    lm, device_map="auto",
                    load_in_8bit=True)
    
    prompt = tok.apply_chat_template( chat, tokenize=False, add_generation_prompt=True)
    inp    = tok(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=256, do_sample=False)
    ans = tok.decode(out[0], skip_special_tokens=True)#.split("assistant")[-1].strip()
    return ans, ctx_nodes


# ═════════════════════ build & serve ═════════════════════════════════════
def build(src: str, *, roles: Sequence[str], srl: SRL_MODEL, kg: KnowledgeGraph):
    text = (wikipedia.page(input("Wikipedia title: ")).content
            if src.lower() == "w"
            else Path(src).read_text(encoding="utf-8"))
    for i, sent in enumerate(sent_tokenize(text)):
        rel, _, role = srl.inference(sent)
        if role: _populate(sent, rel, role[0], roles, kg, i)
    logging.info("graph built: %d nodes — %d edges",
                 kg.g.number_of_nodes(), kg.g.number_of_edges())


def serve(kg: KnowledgeGraph):
    COLORS = {"relation":"red","ARG0":"limegreen","ARG1":"gold","ARG2":"orange",
              "ARG3":"plum","ARG4":"lightblue","ARGM":"lightgray"}
    style  = (
        [{"selector": f"[role = '{r}']",
          "style": {"background-color": c,
                    "font-size": 10,
                    "text-wrap": "wrap",
                    "text-max-width": 120}}
         for r,c in COLORS.items()] +
        [{"selector": "node",
          "style": {"label":"data(label)",
                    "tooltip-text":"data(full)"}},
         {"selector":"edge",
          "style": {"label":"data(label)",
                    "text-rotation":"autorotate",
                    "text-margin-y":"-6px",
                    "font-size": 9}}]
    )

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Input(id="q", type="text", placeholder="Ask"), html.Button("Run", id="go"),
        html.Pre(id="ans"),
        cyto.Cytoscape(id="graph", elements=kg.elements(), layout={"name":"cose"},
                       style={"width":"100%","height":"800px"}, stylesheet=style),
    ])

    @app.callback(Output("ans","children"), Output("graph","elements"),
                  Input("go","n_clicks"), State("q","value"),
                  prevent_initial_call=True)
    def _qa(_, q):
        if not q: return "", kg.elements()
        ans, ctx = answer_with_ctx(q, kg)
        return ans, kg.elements(ctx)

    app.run()


# ═════════════════════ main ══════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser("SRL‑KG dashboard")
    p.add_argument("model")
    p.add_argument("roles", choices=["UP","NOM"])
    p.add_argument("src")
    p.add_argument("--log", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    args = p.parse_args()
    logging.getLogger().setLevel(args.log)

    dev  = "cuda" if torch.cuda.is_available() else "cpu"
    cfg  = json.loads(Path(f"models/{args.model}.json").read_text())
    cfg["device"] = dev
    srl  = SRL_MODEL(**cfg)
    srl.load_state_dict(torch.load(f"models/{args.model}.pt",
                                   map_location=dev))
    srl.eval()

    roles = UP_ROLES if args.roles == "UP" else NOM_ROLES
    kg    = KnowledgeGraph(LocalRetriever())
    build(args.src, roles=roles, srl=srl, kg=kg)
    serve(kg)


if __name__ == "__main__":
    main()
