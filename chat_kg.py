import argparse
import json
import os
import string
import sys
import difflib
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

import torch
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import TreebankWordTokenizer

from model import SRL_MODEL
from dataloaders.UP_dataloader import roles as UP_ROLES
from dataloaders.NomBank_dataloader import roles as NOM_ROLES

# ───────────────────────────── constants ──────────────────────────────
DEFAULT_REL_THRESHOLD   = 0.5
DEFAULT_ROLE_THRESHOLD  = 0.75
EMBED_MODEL             = 'all-MiniLM-L6-v2'
TOP_K_FACTS             = 5
ENTITY_SIM_THRESHOLD    = 0.85

# ────────────────────────── helper functions ──────────────────────────
def _canonical(txt: str) -> str:
    tbl = str.maketrans('', '', string.punctuation)
    return ' '.join(txt.lower().translate(tbl).split())

# ────────────────────────── data structures ────────────────────────────
@dataclass
class RoleSpan:
    role: str
    text: str
    indices: List[int]
    confidence: float

@dataclass
class Fact:
    subject: str
    predicate: str
    object:  str
    role:     str
    confidence: float
    context: Optional[str] = None

# ────────────────────────── main KG class ─────────────────────────────
class ChatKG:
    def __init__(
        self,
        srl_model: SRL_MODEL,
        roles: List[str],
        kg_path: str,
        llm_model: str,
        auto_confirm: bool,
        rel_threshold: float = DEFAULT_REL_THRESHOLD,
        role_threshold: float = DEFAULT_ROLE_THRESHOLD,
    ):
        self.srl            = srl_model
        self.roles          = roles
        self.kg_path        = kg_path
        self.llm_model      = llm_model
        self.auto_confirm   = auto_confirm
        self.rel_threshold  = rel_threshold
        self.role_threshold = role_threshold

        # ── load existing facts ─────────────────────────────────────────
        if os.path.exists(kg_path):
            with open(kg_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            self.facts = [Fact(**f) for f in data.get('facts', [])]
        else:
            self.facts = []

        # ── canonical entity map ────────────────────────────────────────
        self._canon: Dict[str, str] = {}
        for fact in self.facts:
            for ent in (fact.subject, fact.object):
                self._canon.setdefault(_canonical(ent), ent)

        # ── sentence‐transformer embedder ───────────────────────────────
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # ── build entity index ──────────────────────────────────────────
        self.entity_list = list(self._canon.values())
        if self.entity_list:
            embs = self.embedder.encode(self.entity_list, convert_to_numpy=True)
            faiss.normalize_L2(embs)
            self.entity_index = faiss.IndexFlatIP(embs.shape[1])
            self.entity_index.add(embs)
        else:
            dim = self.embedder.get_sentence_embedding_dimension()
            self.entity_index = faiss.IndexFlatIP(dim)

        # ── build fact index ────────────────────────────────────────────
        self.fact_texts = [f"{f.subject} {f.predicate} {f.object}" for f in self.facts]
        if self.fact_texts:
            fact_embs = self.embedder.encode(self.fact_texts, convert_to_numpy=True)
            faiss.normalize_L2(fact_embs)
            self.fact_index = faiss.IndexFlatIP(fact_embs.shape[1])
            self.fact_index.add(fact_embs)
        else:
            dim = self.embedder.get_sentence_embedding_dimension()
            self.fact_index = faiss.IndexFlatIP(dim)

        # ── adjacency: entity → list of fact‐indices ────────────────────
        self.adj = defaultdict(list)
        for idx, fact in enumerate(self.facts):
            self.adj[fact.subject].append(idx)
            self.adj[fact.object].append(idx)

    def _save(self):
        tmp = self.kg_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump({'facts': [asdict(f) for f in self.facts]},
                      fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.kg_path)

    def _resolve_entity(self, name: str) -> str:
        key = _canonical(name)
        # exact
        if key in self._canon:
            return self._canon[key]

        # semantic lookup via FAISS
        emb = self.embedder.encode([name], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.entity_index.search(emb, k=1)
        if D[0][0] >= ENTITY_SIM_THRESHOLD:
            existing = self.entity_list[I[0][0]]
            if self.auto_confirm or input(f"Use '{existing}' for '{name}'? [Y/n] ") in ('','y','yes'):
                self._canon[key] = existing
                return existing

        # fallback to difflib
        near = difflib.get_close_matches(key, self._canon.keys(), n=1, cutoff=0.8)
        if near:
            existing = self._canon[near[0]]
            if self.auto_confirm or input(f"Use existing '{existing}' for '{name}'? [Y/n] ") in ('','y','yes'):
                self._canon[key] = existing
                return existing

        # new entity
        self._canon[key] = name
        self.entity_list.append(name)
        faiss.normalize_L2(emb)
        self.entity_index.add(emb)
        return name

    def _extract_spans(self, tokens: List[str], role_logits: torch.Tensor) -> List[RoleSpan]:
        probs = torch.sigmoid(role_logits)
        spans: List[RoleSpan] = []
        for idx, role_label in enumerate(self.roles[2:]):
            if idx >= probs.shape[1]:
                break
            token_probs = probs[:, idx]
            idxs = (token_probs > self.role_threshold).nonzero(as_tuple=True)[0].tolist()
            if not idxs:
                continue
            # group contiguous indices
            groups, curr = [], [idxs[0]]
            for i in idxs[1:]:
                if i == curr[-1] + 1:
                    curr.append(i)
                else:
                    groups.append(curr); curr = [i]
            groups.append(curr)
            best = max(groups, key=lambda g: token_probs[g].mean().item())
            text = ' '.join(tokens[i] for i in best)
            spans.append(RoleSpan(role_label, text, best,
                                  token_probs[best].mean().item()))
        return spans

    def _add_fact(self, sentence: str):
        rel_logits, _, role_batches = self.srl.inference(sentence)
        if rel_logits is None or role_batches is None:
            print('[!] SRL model produced no output.')
            return
        tokens   = TreebankWordTokenizer().tokenize(sentence)
        rel_mask = torch.sigmoid(rel_logits) > self.rel_threshold
        rel_positions = rel_mask.nonzero(as_tuple=True)[0].tolist()
        if not rel_positions:
            print('[!] No predicate detected in:', sentence)
            return

        for i, pos in enumerate(rel_positions):
            predicate = tokens[pos]
            spans    = self._extract_spans(tokens, role_batches[0][i])

            subj_span = next((s for s in spans if s.role=='ARG0'), None)
            obj_span  = next((s for s in spans if s.role=='ARG1'), None)
            if not subj_span:
                subj_span = next((s for s in spans if s.role=='ARG1'), None)
                obj_span  = next((s for s in spans if s.role=='ARG2'), None)

            subj = self._resolve_entity(subj_span.text) if subj_span else None
            obj  = self._resolve_entity(obj_span.text)  if obj_span  else None

            # core fact
            if subj and obj:
                conf = float((subj_span.confidence + obj_span.confidence)/2)
                core = Fact(subj, predicate, obj, 'ARG0-ARG1', conf, sentence)
                if core not in self.facts:
                    self.facts.append(core)
                    self._index_new_fact(core)
                    print(f"[+] Added: {subj}-{predicate}->{obj} (conf={conf:.2f})")

            # modifiers
            for span in spans:
                if span.role in ('ARG0','ARG1','ARG2'):
                    continue
                target   = subj or predicate
                pred_lab = f"{predicate}_{span.role}"
                mod_fact = Fact(target, pred_lab, span.text,
                                span.role, span.confidence, sentence)
                if mod_fact not in self.facts:
                    self.facts.append(mod_fact)
                    self._index_new_fact(mod_fact)
                    print(f"[+] Mod: {target}-{pred_lab}->{span.text} (conf={span.confidence:.2f})")

        # answer to user using LLM
        prompt = (
            "You are a helpful assistant. This statement is used to add to memory, resond in a conversational way.\n"
        )
        chat = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': sentence},
        ]
        resp = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=chat
        )
        print(resp.choices[0].message.content.strip())
        
        # save KG
        self._save()

    def _index_new_fact(self, fact: Fact):
        txt = f"{fact.subject} {fact.predicate} {fact.object}"
        emb = self.embedder.encode([txt], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        self.fact_index.add(emb)
        self.fact_texts.append(txt)
        new_idx = len(self.fact_texts) - 1
        self.adj[fact.subject].append(new_idx)
        self.adj[fact.object].append(new_idx)

    def _answer_question(self, question: str):
        # 1) embed + retrieve top-K seed facts
        q_emb = self.embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.fact_index.search(q_emb, k=TOP_K_FACTS)
        seed_idxs = [i for i in I[0] if i < len(self.facts)]
        if not seed_idxs:
            print("I don't know")
            return

        # 2) expand to 1-hop neighbors
        neighbor_idxs = set(seed_idxs)
        for idx in seed_idxs:
            fact = self.facts[idx]
            for ent in (fact.subject, fact.object):
                neighbor_idxs.update(self.adj.get(ent, []))

        final_idxs  = sorted(neighbor_idxs)
        final_facts = [self.fact_texts[i] for i in final_idxs]

        # 3) prompt LLM
        prompt = (
            "You are a helpful assistant. Use ONLY the facts below. "
            "If the answer is not there, reply 'I don't know.' if multiple answers are possible make it clear and ask to be clearer\n\n"
            "Facts:\n" + "\n".join(f"- {f}" for f in final_facts) +
            f"\n\nQuestion: {question}"
        )
        try:
            resp = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{'role': 'system', 'content': prompt}]
            )
            print(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f'[!] OpenAI error: {e}')

    def chat_loop(self):
        print("[ChatKG] Enter statements, ask questions with '?', or 'exit'.")
        while True:
            try:
                line = input('> ').strip()
            except (EOFError, KeyboardInterrupt):
                print('\nGoodbye.')
                break
            if not line:
                continue
            lw = line.lower()
            if lw in ('exit', 'quit'):
                break
            if line.endswith('?') or lw.split()[0] in ('who','what','when','where','why','how'):
                self._answer_question(line)
            else:
                self._add_fact(line)

def main():
    parser = argparse.ArgumentParser('ChatKG builder & QA')
    parser.add_argument('-m','--model', required=True,
                        help='SRL model name (models/<name>.json & .pt)')
    parser.add_argument('-r','--roles', choices=('UP','NOM'), default='UP',
                        help='SRL role set')
    parser.add_argument('-k','--kg-file', default='kg.json',
                        help='path to persist KG')
    parser.add_argument('-l','--llm-model', default='gpt-4o-mini',
                        help='OpenAI LLM model')
    parser.add_argument('-c','--auto-confirm', action='store_true',
                        help='resolve entities without prompt')
    args = parser.parse_args()

    key = os.getenv('OPENAI_API_KEY')
    if not key:
        sys.exit('[!] Set OPENAI_API_KEY')
    openai.api_key = key

    cfg_path = os.path.join('models', f"{args.model}.json")
    if not os.path.exists(cfg_path):
        sys.exit(f'[!] Missing SRL config: {cfg_path}')
    cfg = json.load(open(cfg_path))
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    srl = SRL_MODEL(**cfg)
    srl.load_state_dict(torch.load(
        os.path.join('models', f"{args.model}.pt"),
        map_location=cfg['device']
    ))
    srl.to(cfg['device']).eval()

    roles = UP_ROLES if args.roles == 'UP' else NOM_ROLES
    ChatKG(srl, roles, args.kg_file, args.llm_model, args.auto_confirm).chat_loop()

if __name__ == '__main__':
    main()
