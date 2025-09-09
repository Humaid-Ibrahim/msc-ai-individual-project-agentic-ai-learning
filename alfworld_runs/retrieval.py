from pathlib import Path
import glob, json
import numpy as np
from typing import List, Dict, Any
import re

# --- only external dependency ----------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
# ----------------------------------------------------------------------------



def _load_episodes(exp_dir):
    """returns list[(str first_observation, dict full_json)]"""
    out = []
    for fp in glob.glob(f"{exp_dir}/*.json"):
        j = json.load(open(fp))
        traj = j.get("trajectory", [])
        if traj:                                        # <-- skip empties
            out.append((traj[0]["obs_before"], j))
    if not out:
        raise ValueError(f"All trajectories in {exp_dir} are empty.")
    return out



class ExperienceStore:
    """
    Simple TF-IDF nearest-neighbour retriever over a directory of raw chat-log JSONs.

    * scans exp_dir/*.json
    * for each file, loads the JSON (a list of messages)
    * extracts the first user message's content as the document
    * fits a TF-IDF on all those first‐observations
    * nearest(query, k) returns the full raw trace-lists of the top‐k matches
    """
    def __init__(self, exp_dir: str, ngram_range=(1, 2), min_df=1):
        self.docs = []
        self.exps = []

        # load every JSON in the folder
        for fp in glob.glob(f"{exp_dir}/*.json"):
            with open(fp, 'r') as f:
                trace = json.load(f)

            if not isinstance(trace, list):
                # unexpected format, skip
                continue

            # find first user message
            first_user = next((m for m in trace if m.get("role") == "user"), None)
            if not first_user:
                continue

            first_obs = first_user.get("content", "").strip()
            if not first_obs:
                continue

            self.docs.append(first_obs)
            # keep the entire raw list so we can return it later
            self.exps.append(trace)

        if not self.docs:
            raise ValueError(f"No valid trajectories found in {exp_dir}")

        # build TF-IDF index
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
        )
        self.doc_mat = self.vectorizer.fit_transform(self.docs)

    def nearest(self, query: str, k: int = 4):
        """
        Returns the *k* full trace (list-of-message-dicts) whose first user
        message is most similar to *query* (cosine similarity over TF-IDF).
        """
        q_vec = self.vectorizer.transform([query])             # (1,|V|)
        sims  = cosine_similarity(q_vec, self.doc_mat).ravel() # (N,)
        topk = sims.argsort()[::-1][:k]
        return [self.exps[i] for i in topk]
    


class RuleStoreOld:
    def __init__(self, rule_list, method: str = "tfidf", agent=None):
        """
        method: one of "tfidf", "bm25", or "llm".
        If method="llm", you must pass a lightweight LLM `agent` with an `.act(prompt)` interface.
        """
        self.rules = list(rule_list)
        self.method = method.lower()
        if self.method not in {"tfidf", "bm25", "llm"}:
            raise ValueError("method must be 'tfidf', 'bm25', or 'llm'")

        if self.method == "tfidf":
            # initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            self.rule_mat = self.vectorizer.fit_transform(self.rules)

        elif self.method == "bm25":
            # initialize BM25: tokenize each rule for the corpus
            self.tokenized_rules = [rule.split() for rule in self.rules]
            self.bm25 = BM25Okapi(self.tokenized_rules)

        else:  # llm
            if agent is None:
                raise ValueError("You must provide an 'agent' when using method='llm'")
            self.agent = agent

    def nearest(self, query: str, k: int = 6):
        """
        Returns the top-k rules by similarity according to the selected method.
        - tfidf: cosine similarity on TF-IDF vectors
        - bm25: BM25 Okapi ranking
        - llm: prompt an LLM to rank rules and parse its output
        """
        if self.method == "tfidf":
            q_vec = self.vectorizer.transform([query])            # (1,|V|)
            sims = cosine_similarity(q_vec, self.rule_mat).ravel()
            idxs = sims.argsort()[::-1][:k]
            return [self.rules[i] for i in idxs]

        if self.method == "bm25":
            tokens = query.split()
            scores = np.array(self.bm25.get_scores(tokens))
            idxs = scores.argsort()[::-1][:k]
            return [self.rules[i] for i in idxs]

        # llm-based retrieval
        # build a prompt listing the query and candidate rules
        prompt = f"""
You are a rule-ranking assistant. Given the user query:
\"{query}\"

And the following candidate rules:
"""

        for i, rule in enumerate(self.rules, 1):
            prompt += f"\n{i}. {rule}"
        prompt += (
            f"\n\nPlease rank the top {k} most relevant and helpful rules by returning their numbers in order, comma-separated."
        )
        resp = self.agent.act(prompt).strip()
        # parse numbers from the LLM response
        selected = []
        for part in resp.replace(';', ',').split(','):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(self.rules):
                    selected.append(self.rules[idx])
                    if len(selected) >= k:
                        break
        # fallback to tfidf if parsing fails or no selection
        if not selected:
            return RuleStore(self.rules, method="tfidf").nearest(query, k)
        return selected



#     def assistance_nearest(self, query: str, k: int = 6):
#         if self.method == "tfidf":
#             q_vec = self.vectorizer.transform([query])            # (1,|V|)
#             sims = cosine_similarity(q_vec, self.rule_mat).ravel()
#             idxs = sims.argsort()[::-1][:k]
#             return [self.rules[i] for i in idxs]

#         if self.method == "bm25":
#             tokens = query.split()
#             scores = np.array(self.bm25.get_scores(tokens))
#             idxs = scores.argsort()[::-1][:k]
#             return [self.rules[i] for i in idxs]

#         # llm-based retrieval
#         # build a prompt listing the query and candidate rules
#         prompt = f"""
# You are a rule-ranking assistant. Given the current task and state:
# \"{query}\"

# And the following candidate rules:
# """

#         for i, rule in enumerate(self.rules, 1):
#             prompt += f"\n{i}. {rule}"
#         prompt += (
#             f"\n\nPlease find the most relevant and helpful rule that would help the agent. If there aren't any relevant hints please strictly output NONE."
#         )
#         resp = self.agent.act(prompt).strip()
#         # parse numbers from the LLM response
#         selected = []
#         for part in resp.replace(';', ',').split(','):
#             part = part.strip()
#             if part.isdigit():
#                 idx = int(part) - 1
#                 if 0 <= idx < len(self.rules):
#                     selected.append(self.rules[idx])
#                     if len(selected) >= k:
#                         break
#         # fallback to tfidf if parsing fails or no selection
#         if not selected:
#             return RuleStore(self.rules, method="tfidf").nearest(query, k)
#         return selected

    def assistance_nearest(self, query: str, k: int = 6, shortlist: int = 15):
        """
        Return up to k rules relevant to `query`.
        - tfidf/bm25: unchanged.
        - llm: build a small shortlist via bm25/tfidf, ask LLM to pick 1-2 (or NONE),
               then pad from shortlist to reach k; fallback to tfidf on parse failure.
        """
        # --- classic branches unchanged ---
        if self.method == "tfidf":
            q_vec = self.vectorizer.transform([query])            # (1,|V|)
            sims = cosine_similarity(q_vec, self.rule_mat).ravel()
            idxs = sims.argsort()[::-1][:k]
            return [self.rules[i] for i in idxs]
    
        if self.method == "bm25":
            tokens = query.split()
            scores = np.array(self.bm25.get_scores(tokens))
            idxs = scores.argsort()[::-1][:k]
            return [self.rules[i] for i in idxs]
    
        # -------------------------------
        # LLM-based retrieval (improved)
        # -------------------------------
        import re
    
        def _uniq(seq):
            seen = set(); out = []
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out
    

        # # 1) Build a SMALL candidate list using BM25 (preferred) or TF-IDF
        # if hasattr(self, "bm25") and self.bm25 is not None:
        #     tokens = query.split()
        #     scores = np.array(self.bm25.get_scores(tokens))
        #     cand_idxs = scores.argsort()[::-1][:min(shortlist, len(self.rules))]
        # elif hasattr(self, "vectorizer") and hasattr(self, "rule_mat"):
        #     q_vec = self.vectorizer.transform([query])
        #     sims = cosine_similarity(q_vec, self.rule_mat).ravel()
        #     cand_idxs = sims.argsort()[::-1][:min(shortlist, len(self.rules))]
        # else:
        #     # last resort: first N rules
        #     cand_idxs = np.arange(min(shortlist, len(self.rules)))
    
        # cand_rules = [self.rules[i] for i in cand_idxs]

        cand_rules = self.rules
            
        # 2) Build a tight 1.5B-friendly prompt (pick ONE or NONE; optionally allow 2)
        prompt = (
            "You are a small hint assistant. Choose the ONE hint that will help the household agent at this current state.\n"
            "If the agent is repeating itself too many times, provide the most relevant hint. Otherwise, output NONE\n\n"
            "Make sure to view the current location, current inventory, thought, and task goal before providing hints to avoid misleading information."
            f"===== Task & state =====\n{query}\n\n"
            "===== Hints List =====\n"
        )
        # prompt = (
        #     "You are a small rule picker. Choose the ONE rule that helps the agent NOW.\n"
        #     "Prefer rules with a concrete next action (open, go to, take, put, etc.).\n\n"
        #     f"Task & state:\n{query}\n\n"
        #     "Candidate rules:\n"
        # )
        for i, r in enumerate(cand_rules, 1):
            prompt += f"{i}) {r}\n"
        prompt += (
            "\nReply in this exact format:\n"
            "ANSWER: <index or NONE>\n"
            "Do not explain."
        )
    
        # 3) Ask the LLM
        try:
            resp = self.agent.act(prompt).strip()
        except Exception:
            # fallback immediately
            return RuleStore(self.rules, method="tfidf").nearest(query, k)
    
        # 4) Parse robustly: accept "ANSWER: [2]", "ANSWER: 2", "2, 3", or "NONE"
        lower = resp.lower()
        if "none" in lower:
            picked_local_idxs = []
        else:
            nums = re.findall(r"\b\d+\b", resp)
            picked_local_idxs = [int(n) - 1 for n in nums if 1 <= int(n) <= len(cand_rules)]
    
        # Map back to global rule strings; remove near-duplicates
        selected = []
        for li in picked_local_idxs:
            r = cand_rules[li]
            selected.append(r)
            if len(selected) >= k:
                break

    
        # 6) Final safety: fallback to tfidf if nothing selected
        if not selected:
            return RuleStore(self.rules, method="tfidf").nearest(query, k)
    
        return selected[:k]


class RuleStore:
    def __init__(self, rule_list, method: str = "tfidf", agent=None):
        """
        method: one of "tfidf", "bm25", or "llm".
        If method="llm", you must pass a lightweight LLM `agent` with an `.act(prompt)` interface.
        """
        # rule_list = "/workspace/stateact-ft/alfworld_runs/game_logs/hints_by_type.json"
        # with open(rule_list, "r") as f:
        self.rules = rule_list
            
        self.method = method.lower()
        if self.method not in {"tfidf", "bm25", "llm"}:
            raise ValueError("method must be 'tfidf', 'bm25', or 'llm'")

        if self.method == "tfidf":
            # initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            self.rule_mat = self.vectorizer.fit_transform(self.rules)

        elif self.method == "bm25":
            # initialize BM25: tokenize each rule for the corpus
            self.tokenized_rules = [rule.split() for rule in self.rules]
            self.bm25 = BM25Okapi(self.tokenized_rules)

        else:  # llm
            if agent is None:
                raise ValueError("You must provide an 'agent' when using method='llm'")
            self.agent = agent


    
    def nearest(self, query: str, k: int = 3, env_type: str = "cool"):
        """
        Return up to k rules relevant to `query`.
        - tfidf/bm25: unchanged behaviour, but return from cand_rules.
        - llm: ask for up to k indices (JSON array), then pad from full list.
        """
        cand_rules = self.rules[env_type]["hints"]  # list[str]
        token_count_dictionary = {"in_token_all":0, "in_token_message":0, "out_token_action":0}
        
        # --- classic branches (return from cand_rules) ---
        if self.method == "tfidf":
            q_vec = self.vectorizer.transform([query])                # (1, |V|)
            sims = cosine_similarity(q_vec, self.rule_mat).ravel()    # assumes rule_mat matches cand_rules
            idxs = sims.argsort()[::-1][:k]
            return [cand_rules[i] for i in idxs], token_count_dictionary
    
        if self.method == "bm25":
            tokens = query.split()
            scores = np.array(self.bm25.get_scores(tokens))
            idxs = scores.argsort()[::-1][:k]
            return [cand_rules[i] for i in idxs], token_count_dictionary
    
        # LLM-based retrieval
        prompt = (
            f"You are selecting helpful hints for a household agent.\n"
            f"Choose up to {k} DISTINCT hints that are immediately useful for the current state.\n"
            "If none apply, return an empty list.\n\n"
            "Read the current goal, location, inventory and thought to avoid redundant/misleading hints.\n\n"
            "===== Task & state =====\n"
            f"{query}\n\n"
            "===== Candidate hints (numbered) =====\n"
        )
        for i, r in enumerate(cand_rules, 1):
            prompt += f"{i}) {r}\n"
    
        prompt += (
            '\nReturn STRICT JSON only (no prose):\n'
            '{"answer": [<indices from the list above>]}\n'
            "Do not include anything else."
        )
    
        # Call the model
        try:
            resp, token_count_dictionary = self.agent.act(prompt, return_token_count=True)
            resp = resp.strip()
        except Exception:
            # simple fallback → TF-IDF over full list (no shortlist)
            if hasattr(self, "vectorizer") and hasattr(self, "rule_mat"):
                q_vec = self.vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.rule_mat).ravel()
                idxs = sims.argsort()[::-1][:k]
                return [cand_rules[i] for i in idxs]
            # last resort: first k
            return cand_rules[:k], token_count_dictionary
    
        # Parse JSON robustly
        picked_local = []
        try:
            data = json.loads(resp)
            if isinstance(data, dict) and isinstance(data.get("answer"), list):
                picked_local = [int(x) for x in data["answer"] if isinstance(x, (int, str))]
        except Exception:
            # fallback: pull numbers like "2, 5"
            picked_local = [int(n) for n in re.findall(r"\b\d+\b", resp)]
    
        # Map to zero-based, unique, in-range
        selected_idxs = []
        for n in picked_local:
            li = int(n) - 1
            if 0 <= li < len(cand_rules) and li not in selected_idxs:
                selected_idxs.append(li)
            if len(selected_idxs) >= k:
                break
    
        # If model returned none, fallback gently (no shortlist)
        if not selected_idxs:
            if hasattr(self, "vectorizer") and hasattr(self, "rule_mat"):
                q_vec = self.vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.rule_mat).ravel()
                idxs = sims.argsort()[::-1][:k]
                return [cand_rules[i] for i in idxs]
            return cand_rules[:k], token_count_dictionary
    
        # Pad from full list (in order) to reach k
        if len(selected_idxs) < k:
            for i in range(len(cand_rules)):
                if i not in selected_idxs:
                    selected_idxs.append(i)
                    if len(selected_idxs) >= k:
                        break
    
        return [cand_rules[i] for i in selected_idxs[:k]], token_count_dictionary


class FacetRetriever:
    def __init__(self, facet_json_path: str, agent=None, ngram_range=(1,2), min_df=1):
        self.agent = agent
        self.envs = self._load_facets(facet_json_path)

        # TF-IDF over goal-success (for top-M env selection)
        self.goal_docs = [" \n".join(env["goal_success"]) for env in self.envs]
        self.goal_vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, lowercase=True)
        self.goal_mat = self.goal_vectorizer.fit_transform(self.goal_docs)

    def _load_facets(self, path: str) -> List[Dict[str, Any]]:
        j = json.load(open(path))
        envs = []
        for env_id, blob in j.items():
            envs.append({
                "id": env_id,
                "goal_success": blob.get("goal-success", []) or [],
                "precondition": blob.get("precondition", []) or [],
                "best_practice": blob.get("best-practice", []) or [],
            })
        if not envs:
            raise ValueError(f"No environments found in {path}")
        return envs

    def retrieve_hints(self, query: str, k: int = 6, top_m_envs: int = 1, max_per_env: int = 4) -> List[str]:
        # 1) nearest envs by goal-success
        qv = self.goal_vectorizer.transform([query])
        sims = cosine_similarity(qv, self.goal_mat).ravel()
        top_env_idxs = sims.argsort()[::-1][:top_m_envs]

        # 2) collect precondition + best-practice lines from those envs
        pool, seen = [], set()
        for idx in top_env_idxs:
            env = self.envs[idx]
            for s in (env["precondition"][:max_per_env] + env["best_practice"][:max_per_env]):
                s = (s or "").strip()
                if s and s not in seen:
                    seen.add(s)
                    pool.append(s)

        if not pool:
            return []

        # 3) if LLM is available, enumerate + ask for numbers; else TF-IDF rank fallback
        if self.agent is not None:
            prompt = f'You are a household assistant. Given the task:\n"{query}"\n\nAnd the following hints:\n'
            for i, rule in enumerate(pool, 1):
                prompt += f"\n{i}. {rule}"
            prompt += f"\n\nPlease rank the top {k} most relevant and helpful rules by returning their numbers in order, comma-separated."
            try:
                resp = self.agent.act(prompt).strip()
                idxs = []
                for part in resp.replace(";", ",").split(","):
                    tok = part.strip()
                    if tok.isdigit():
                        j = int(tok) - 1
                        if 0 <= j < len(pool):
                            idxs.append(j)
                            if len(idxs) >= k:
                                break
                if idxs:
                    return [pool[j] for j in idxs]
            except Exception:
                pass

        # TF-IDF fallback over the hint pool
        v = TfidfVectorizer(ngram_range=(1,2), lowercase=True, min_df=1)
        mat = v.fit_transform(pool)
        q = v.transform([query])
        order = cosine_similarity(q, mat).ravel().argsort()[::-1][:k]
        return [pool[i] for i in order]


# --- 2) Tiny adapter so it “looks like” your RuleStore ----------------------
class FacetRuleAdapter:
    def __init__(self, facet_retriever: FacetRetriever):
        self.facet = facet_retriever

    def nearest(self, query: str, k: int = 6):
        # returns List[str] of hints → exactly what build_expel_prompt expects
        return self.facet.retrieve_hints(query, k=k)









# --- Structured rules store (for rules_*.json we generated) ---
from sentence_transformers import SentenceTransformer
import numpy as np

class StructuredRuleStore:
    def __init__(self, path, embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        with open(path) as f:
            self.rules = json.load(f)  # list[dict] with keys: rule, trigger, action_template, tags, env_type, pattern
        # build corpus strings
        self.texts = []
        for r in self.rules:
            parts = [
                r.get("trigger", ""),
                r.get("rule", ""),
                r.get("action_template", ""),
                " ".join(r.get("tags", [])),
                r.get("env_type", ""),
                r.get("pattern", ""),
            ]
            self.texts.append(" || ".join(p for p in parts if p))
        self.model = SentenceTransformer(embed_model)
        self.emb = self.model.encode(self.texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def nearest(self, query, k= 4, env_type=None):
        # optional env filter
        idxs = list(range(len(self.rules)))
        if env_type:
            idxs = [i for i, r in enumerate(self.rules) if r.get("env_type", "") in ("", env_type)]
            if not idxs:  # fallback to all
                idxs = list(range(len(self.rules)))

        qv = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        sub = self.emb[idxs]
        sims = sub @ qv
        topk_local = np.argsort(-sims)[:k]
        top_idxs = [idxs[i] for i in topk_local]
        return [self.rules[i] for i in top_idxs]
