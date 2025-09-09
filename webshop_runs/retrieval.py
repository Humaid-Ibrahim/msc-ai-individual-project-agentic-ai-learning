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
    

class RuleStore:
    def __init__(self, rule_list, method: str = "llm", agent=None, allowed_envs=None):
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

        # ---- normalize rules into env->list[str] ----
        if isinstance(rule_list, dict):
            self.rules_by_env = {str(k).lower(): list(v) for k, v in rule_list.items()}
        else:
            raise TypeError("rule_list must be dict[str, list[str]]")

        self.allowed_envs = ["fashion", "beauty", "food", "furniture", "electronics"]
        
        self.env_docs = {env: "\n".join(hints) for env, hints in self.rules_by_env.items()}
        # initialize backends
        if self.method == "tfidf":
            # Vectorize per-env docs for category selection
            self.cat_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            self.cat_mat = self.cat_vectorizer.fit_transform(
                [self.env_docs[e] for e in self.allowed_envs]
            )  # shape: (E, V)

            # Vectorize all hints within each env for retrieval
            self.env_vectorizers = {}
            self.env_mats = {}
            for env in self.allowed_envs:
                vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
                mat = vec.fit_transform(self.rules_by_env.get(env, []))  # (N_env, V_env)
                self.env_vectorizers[env] = vec
                self.env_mats[env] = mat

        elif self.method == "bm25":
            if BM25Okapi is None:
                raise ImportError("rank_bm25 not installed. `pip install rank-bm25`")
            # tokenized per-env docs for category selection
            self.env_tokens = {env: self._simple_tok(doc) for env, doc in self.env_docs.items()}
            self.env_bm25 = BM25Okapi([self.env_tokens[e] for e in self.allowed_envs])

            # tokenized hints per env for retrieval
            self.env_bm25_rules = {}
            self.env_tokenized_rules = {}
            for env in self.allowed_envs:
                toks = [self._simple_tok(r) for r in self.rules_by_env.get(env, [])]
                self.env_tokenized_rules[env] = toks
                self.env_bm25_rules[env] = BM25Okapi(toks) if toks else None

        else:  # llm
            if agent is None:
                raise ValueError("You must provide an 'agent' when using method='llm'")
            self.agent = agent

    @staticmethod
    def _simple_tok(text: str) -> List[str]:
        # very light tokenizer suitable for BM25
        return re.findall(r"[a-z0-9]+", text.lower())

    def _choose_env_tfidf(self, query: str) -> str:
        if not self.allowed_envs:
            return "fashion"
        qv = self.cat_vectorizer.transform([query])  # (1,V)
        sims = (self.cat_mat @ qv.T).toarray().ravel()  # cosine w/ L2-normalized TF-IDF not guaranteed; still fine for ranking
        idx = int(np.argmax(sims))
        return self.allowed_envs[idx]

    def _choose_env_bm25(self, query: str) -> str:
        if not self.allowed_envs:
            return "fashion"
        q = self._simple_tok(query)
        # rank per-env docs (BM25 over category docs)
        scores = self.env_bm25.get_scores(q)
        idx = int(np.argmax(scores))
        return self.allowed_envs[idx]


    def nearest(self, query: str, k: int = 3, env_type=None):
        """
        Return up to k rules relevant to `query`.
        - tfidf/bm25: unchanged behaviour, but return from cand_rules.
        - llm: ask for up to k indices (JSON array), then pad from full list.
        """

        # infer env_type from the query
        
        allowed_envs = self.allowed_envs 
        chosen_env = env_type
        episode_tokens = {"total_in_token_accumulated":0,
                         "total_in_token_message_accumulated":0,
                          "total_out_token_accumulated":0}     
        
        if env_type is None or str(env_type).lower() == "auto":
            if self.method == "llm":
                # print('auto finding prompt...')
                cls_prompt = (
                    "You are a strict classifier. Classify the user's shopping query "
                    f"into EXACTLY ONE of the following environments: {allowed_envs}.\n\n"
                    "Return STRICT JSON ONLY (no prose), with this schema:\n"
                    '{"env_type": "<one of the allowed values>"}\n\n'
                    "Query:\n"
                    f"{query}\n"
                )
                try:
                    # print(cls_prompt)
                    resp = self.agent(cls_prompt)
                    tokens_cat = resp[1]
                    resp = resp[0].strip()
                    # print(resp)
                    picked = None
                    # Try JSON first
                    try:
                        data = json.loads(resp)
                        if isinstance(data, dict) and isinstance(data.get("env_type"), str):
                            picked = data["env_type"].strip().lower()
                        # print('model predicted category: ', picked)
                    except Exception as e:
                        # print("ERROR: ", e)
                        # Fallback: pick first allowed label mentioned in the text
                        m = re.search(r"\b(fashion|beauty|food|furniture|electronics)\b", resp, re.I)
                        if m:
                            picked = m.group(1).lower()
    
                    if picked in allowed_envs:
                        chosen_env = picked
                    else:
                        # last resort: default to "fashion" if classifier failed
                        chosen_env = "fashion"
                except Exception:
                    # If classification call itself fails, fall back gracefully
                    chosen_env = "fashion"
                    
            elif self.method == "tfidf":
                chosen_env = self._choose_env_tfidf(query)
            elif self.method == "bm25":
                chosen_env = self._choose_env_bm25(query)
            else:
                chosen_env = "fashion"
        # -------------------------------

        cand_rules = self.rules[chosen_env]["hints"]  # list[str]
        
        if k == 0: # Use all hints if k=0
            return cand_rules, episode_tokens
            
        # # --- classic branches (return from cand_rules) ---
        # if self.method == "tfidf":
        #     q_vec = self.vectorizer.transform([query])                # (1, |V|)
        #     sims = cosine_similarity(q_vec, self.rule_mat).ravel()    # assumes rule_mat matches cand_rules
        #     idxs = sims.argsort()[::-1][:k]
        #     return [cand_rules[i] for i in idxs], episode_tokens
    
        # if self.method == "bm25":
        #     tokens = query.split()
        #     scores = np.array(self.bm25.get_scores(tokens))
        #     idxs = scores.argsort()[::-1][:k]
        #     return [cand_rules[i] for i in idxs], episode_tokens

        if self.method == "tfidf":
            vec = self.env_vectorizers.get(chosen_env)
            mat = self.env_mats.get(chosen_env)
            if vec is None or mat is None or mat.shape[0] == 0:
                return [], {"env_type": chosen_env}
            qv = vec.transform([query])  # (1,V_env)
            scores = (mat @ qv.T).toarray().ravel()  # (N_env,)
            top_idx = np.argsort(-scores)[:k]
            return [cand_rules[i] for i in top_idx], episode_tokens

        elif self.method == "bm25":
            bm = self.env_bm25_rules.get(chosen_env)
            toks = self.env_tokenized_rules.get(chosen_env, [])
            if bm is None or len(toks) == 0:
                return [], {"env_type": chosen_env}
            q = self._simple_tok(query)
            scores = bm.get_scores(q)  # (N_env,)
            top_idx = np.argsort(-scores)[:k]
            return [cand_rules[i] for i in top_idx], episode_tokens
        
        # LLM-based retrieval
        prompt = (
            f"You are selecting helpful hints for a shopping agent.\n"
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
            # resp = self.agent(prompt)[0].strip()
            resp = self.agent(prompt)
            tokens_retrieve = resp[1]
            resp = resp[0].strip()
            # print("model response:", resp)
        except Exception as e:
            print("error for some reason")
            print(e)
            # simple fallback → TF-IDF over full list (no shortlist)
            if hasattr(self, "vectorizer") and hasattr(self, "rule_mat"):
                q_vec = self.vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.rule_mat).ravel()
                idxs = sims.argsort()[::-1][:k]
                return [cand_rules[i] for i in idxs]
            # last resort: first k
            return cand_rules[:k], episode_tokens
    
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
            print("Model failed to retrieve")
            if hasattr(self, "vectorizer") and hasattr(self, "rule_mat"):
                q_vec = self.vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.rule_mat).ravel()
                idxs = sims.argsort()[::-1][:k]
                return [cand_rules[i] for i in idxs]
            return cand_rules[:k], episode_tokens
    
        # # Pad from full list (in order) to reach k
        # if len(selected_idxs) < k:
        #     for i in range(len(cand_rules)):
        #         if i not in selected_idxs:
        #             selected_idxs.append(i)
        #             if len(selected_idxs) >= k:
        #                 break

        episode_tokens["total_in_token_accumulated"] = int(tokens_cat.get("prompt_tokens", 0))
        episode_tokens["total_in_token_message_accumulated"] = int(tokens_cat.get("prompt_tokens", 0))
        episode_tokens["total_out_token_accumulated"] = int(tokens_cat.get("completion_tokens", 0))
        
        episode_tokens["total_in_token_accumulated"] += int(tokens_retrieve.get("prompt_tokens", 0))
        episode_tokens["total_in_token_message_accumulated"] += int(tokens_retrieve.get("prompt_tokens", 0))
        episode_tokens["total_out_token_accumulated"] += int(tokens_retrieve.get("completion_tokens", 0))
        
        # return [cand_rules[i] for i in selected_idxs[:k]], episode_tokens
        return [cand_rules[i] for i in selected_idxs], episode_tokens



