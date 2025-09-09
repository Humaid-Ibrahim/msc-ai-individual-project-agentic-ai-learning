#!/usr/bin/env python
"""
extract_hints.py – Extract structured, runtime-usable hints from FAILED trajectories
by aligning them with the environment's ground-truth PDDL plan in traj_data.json.

Add-on: supports GPT via OpenAI API with --llm_type GPTChat (or gptchat)
"""

from __future__ import annotations
import argparse, json, os, pathlib, re, sys, random, time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from tqdm.auto import tqdm
import pandas as pd

# ----------------- GPT (OpenAI) MODEL -------------------
# Minimal wrapper around OpenAI's Chat Completions API
class GPTChat:
    def __init__(self,
                 model: str = "gpt-4o",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 organization: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 600,
                 system_prompt: str = "You are a precise assistant. Reply exactly as requested.",
                 **kwargs):
        # Lazy import to avoid dependency if unused
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "OpenAI python client not installed. `pip install openai`"
            ) from e
            
        # Instantiate client (env vars also work)
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
            organization=organization or os.getenv("OPENAI_ORG") or None,
        )
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.system_prompt = system_prompt

        # Whether to request JSON mode (works on 4o/4o-mini)
        self._use_json_mode = True

    def act(self, prompt: str) -> str:
        # Try JSON mode first; if it fails, retry without it.
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"} if self._use_json_mode else None,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception:
            # Retry once without forcing JSON
            try:
                self._use_json_mode = False
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp.choices[0].message.content or ""
            except Exception as e2:
                raise

# ----------------- ENV / MISC -----------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TRITON_DISABLE", "1")

AGENT_MODEL_MAPPING = {
    "VLLMChat": [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
    ],
    "GPTChat": [
        # Add/override as you like
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ],
}



PATTERN = re.compile(r'Instruction:\s*(.*?)\s*\[Search\]', re.DOTALL)

def _extract_from_text(text: str) -> List[str]:
    return [m.strip() for m in PATTERN.findall(text)]

def extract_instruction(obj: Union[str, List[Dict[str, Any]]]) -> str:
    """
    Extract the final instruction text between 'Instruction:' and '[Search]'.
    Prefers matches found in messages where role == 'user'.
    Returns '' if nothing is found.
    """
    messages = None

    # If given a string that looks like a Python list of dicts, parse it.
    if isinstance(obj, str):
        try:
            parsed = ast.literal_eval(obj)
            if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                messages = parsed
        except Exception:
            pass

    # If we have a parsed message list
    if isinstance(messages, list):
        hits = []
        # 1) Search only user messages first (to avoid the example in system content)
        for msg in messages:
            if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
                hits.extend(_extract_from_text(msg['content']))
        if hits:
            return hits[-1]  # last occurrence in user messages

        # 2) Fallback: search all message contents
        all_text = "\n".join(str(m.get('content', '')) for m in messages)
        all_hits = _extract_from_text(all_text)
        return all_hits[-1] if all_hits else ""

    # If we couldn’t parse it as a list, treat it as raw text
    hits = _extract_from_text(str(obj))
    return hits[-1] if hits else ""




def get_agent_and_model(llm_type,
                        temperature=0.0,
                        proposed_model="",
                        force_model=False,
                        max_tokens=800,
                        openai_api_key: Optional[str]=None,
                        openai_base_url: Optional[str]=None,
                        openai_org: Optional[str]=None,
                        **kwargs):
    """Return (agent, model_str) for either local VLLM or GPTChat."""
    lt = (llm_type or "").strip()
    lt_norm = lt.lower()

    # ---------- GPT branch ----------
    if lt_norm in {"gptchat", "gpt", "openai"}:
        default = AGENT_MODEL_MAPPING["GPTChat"][0]
        model = proposed_model or default
        if proposed_model and proposed_model not in AGENT_MODEL_MAPPING["GPTChat"]:
            print(f"[WARN] Proposed OpenAI model '{proposed_model}' not in known list; using anyway." if force_model else
                  f"[WARN] Proposed OpenAI model '{proposed_model}' not in known list; falling back to {default}.")
            if not force_model:
                model = default
        openai_api_key = "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"
        agent = GPTChat(
            model=model,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            base_url=openai_base_url or os.getenv("OPENAI_BASE_URL"),
            organization=openai_org or os.getenv("OPENAI_ORG"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return agent, model

    # ---------- Local VLLM branch ----------
    if lt_norm in {"vllmchat", "vllm"}:
        if VLLMChat is None:
            raise RuntimeError("VLLMChat unavailable. Install/ensure LLamp local class is importable.")
        default = AGENT_MODEL_MAPPING["VLLMChat"][0]
        model = proposed_model or default
        if proposed_model and proposed_model not in AGENT_MODEL_MAPPING["VLLMChat"]:
            print("WARNING: Proposed Model is not in LIST!!!")
            if force_model:
                print("WARNING: Using your model in any case.")
            else:
                model = default
        agent = VLLMChat(model=model, temperature=temperature, **kwargs)
        return agent, model

    raise ValueError(f"Unknown llm_type='{llm_type}'. Use 'GPTChat' or 'VLLMChat'.")

# ---------- Prompting ----------

def build_hint_prompt(category: str,
                      product_category: str,
                      goal: str,
                      gold_plan,
                      reward,
                      my_plan) -> str:
    # print(my_plan)
    # goal_txt = goal or "(goal text unavailable)"
    # plan_txt = "\n".join(f"- {s}" for s in gold_plan) if gold_plan else "(no gold plan found)"

#     return f"""
# You are diagnosing why a WebShop shopping assistant failed to purchase the requested item and creating runtime HINTS to avoid future failures in similar tasks. 
# The assistant is only permitted to click/interact with buttons in square brackets [], and has 15 steps to achieve its goal.

# Emit STRICT JSON with this schema:
# {{
#   "hints": [
#     {{
#       "category": "{category}",
#       "text": "≤120 chars, imperative advice the agent can follow in future for similar tasks"
#     }}
#   ]
# }}

# Guidelines:
# - Focus on errors that explain THIS failure; provide hints to avoid failures on SIMILAR tasks in the same environment.
# - Make it generally applicable.
# - Use placeholders like {{item}}, {{size}}, {{color}}, {{attribute}}, {{material}} instead of names/IDs.
# - 1–3 high-value hints max. No duplicates. No meta commentary.
# - JSON only. No extra text.

# Here is the situation.

# Task instructions: {goal}

# =======
# Steps leading to failure (action → observation):
# {my_plan}
# =======
# """
    
#     return f"""
# You are diagnosing why a WebShop shopping agent failed and creating runtime HINTS to avoid future failures in purchasing similar items.

# Emit STRICT JSON with this schema:
# {{
#   "hints": [
#     {{
#       "category": \"{category}",
#       "text": "≤120 chars, imperative advice the agent can follow in future for similar environment types"
#     }}
#   ]
# }}


# Guidelines:
# - The agent can only search and click buttons that appear in the environment.
# - **Always prioritize the task goal constraints** (e.g., {{price}}, {{color}}, {{size}}, {{brand}}, {{item}}) over any behavior shown in the reference.
# - Focus on **concrete, actionable fixes** that generalize across similar tasks: query design, click choice, variant selection.
# - Make hints **generalized** with placeholders like {{color}}, {{size}}, {{price}}, {{item}}, {{price}}. No ASINs, page numbers, item names, price, and descriptions.
# - Do NOT mention the trace, “agent,” or “observation.” Speak as instructions.
# - Keep hints under ≤120 chars.
# - Output only **1–3 non-duplicate, high-value** hints.
# - JSON only. No extra text.

# Internal checklist (think silently; do not output this):
# - Compare goal vs. chosen product(s): which constraint(s) were missed? ({{price}}, {{attribute}}, {{material}}, etc.)
# - What search strategy would've helped avoid the issue?

# Here is the situation.

# Item category: {category}
# Task goal: {goal}

# =======
# The agent's trace:
# {my_plan}
# =======
# """.strip()

#     return f"""
# You are diagnosing why a shopping assistant failed and creating runtime HINTS to avoid future failures in similar tasks.

# Item category (not provided to agent): {category}
# Task goal: {goal}

# =======
# Steps leading to failure (action → observation):
# {my_plan}
# =======

# Emit STRICT JSON with this schema:
# {{
#   "hints": [
#     {{
#       "env_type": "{category}",
#       "text": "≤120 chars, imperative advice the agent can follow in future for similar environment types"
#     }}
#   ]
# }}

# Rules:
# - Focus on errors that explain THIS failure; provide hints that generalize to SIMILAR tasks.
# - Always respect the task goal (e.g., {{object}}, {{container}}, {{location}}, {{item}}, {{price}}, {{color}}, {{size}}, {{brand}}) without adding additional constraints than stated.
# - Use placeholders like {{object}}, {{container}}, {{location}}, {{item}}, {{price}}, {{color}}, {{size}}, {{brand}} instead of IDs, product names, or positions.
# - Emphasize query design and fallback strategies; avoid page-specific or step-by-step instructions.
# - 1–3 high-value hints max. No duplicates. No meta commentary.
# - JSON only. No extra text.
# """.strip()
    
    return f"""
You are diagnosing why a WebShop shopping agent failed to meet the customer requirements and creating runtime HINTS to avoid future failures in purchasing similar items.

Product Category (not provided to agent): {category}
Product Requirements: {goal}

=======
Steps before failure (action → observation):
{my_plan}
=======

Emit STRICT JSON with this schema:
{{
  "hints": [
    {{
      "category": \"{category}",
      "text": "≤120 chars, imperative advice the agent can follow in future for similar environment types"
    }}
  ]
}}

Rules:
- Focus on errors that explain THIS failure; provide hints to avoid failures on SIMILAR tasks.
- Make it generally applicable.
- Use placeholders like {{item}}, {{size}}, {{color}}, {{attribute}}, {{material}} instead of names/IDs.
- 1–2 high-value hints max. No duplicates. No meta commentary.
- JSON only. No extra text.
""".strip()

JSON_BLOCK_RE = re.compile(r"\{.*\}\s*$", re.S)

def parse_model_json(resp: str) -> Optional[Dict[str, Any]]:
    m = JSON_BLOCK_RE.search(resp.strip())
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ---------- I/O helpers ----------
def load_processed_keys(ndjson_path: str) -> set[Tuple[int,str]]:
    done = set()
    p = pathlib.Path(ndjson_path)
    if not p.exists():
        return done
    with open(p) as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add((int(obj["env_idx"]), str(obj.get("trace_file",""))))
            except Exception:
                continue
    return done

def append_ndjson(path: str, obj: Dict[str, Any]) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

def keep_from_instruction(s: str) -> str:
    i = s.lower().find("webshop")
    # print(i)
    return s[i:] if i != -1 else s

# ---------- Main pipeline ----------
def main(args) -> None:
    df = pd.read_csv(args.csv+"webshop_scores.csv")
    df = df[df.total_reward<1]
    # df = df[df.num_of_steps<15]
    # df = df[df.total_reward<=0.5]
    df.reset_index(inplace=True, drop=True)
    print(f"Generating hints from {len(df)} trajectories...")
    
    train_metadata = pd.read_csv("data/train_metadata.csv")
    
    agent, actual_model = get_agent_and_model(
        llm_type=args.llm_type,
        temperature=args.temperature,
        proposed_model=args.model,
        force_model=args.force_model,
        max_tokens=args.max_tokens,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        openai_org=args.openai_org,
        max_model_len=args.max_model_len,   # ignored by GPTChat
        quantization=bool(args.quantization),  # ignored by GPTChat
        tensor_parallel_size=args.gpus,        # ignored by GPTChat
        seed=args.seed,
    )
    
    print(f"[INFO] using model = {actual_model}")

    already = load_processed_keys(args.out)
    print(f"[INFO] {len(already)} items already in {args.out}; will skip those.")


    kept = 0; skipped = 0; all_hint_texts = Counter()
    pbar = tqdm(total=len(df), desc="Extracting hints", unit="env")
    marker = "WebShop [SEP] Instruction: [SEP]"

    for i, row in df.iterrows():
        env_idx  = i
        category = str(train_metadata["category"][i])
        # category = str(row.get("category","unknown"))
        if category == "grocery":
            category = "food"
        if category == "garden":
            category = "furniture"
        category_norm = category.strip().lower()  # NEW
            
        # product_category = str(row.get("product_category","unknown"))
        # product_category = str(row.get("product_category","unknown"))
        product_category = str(train_metadata["product_category"][i])

        with open(args.csv+str(row.get("trace_file"))) as f:
            my_plan = json.load(f)

        # print(my_plan)
        
        goal = extract_instruction(str(my_plan))
        
        # s = str(row.get("expert_path"))
        # gold_plan = s[s.find(marker):] if marker in s else s
        # gold_plan = str(row.get("expert_summary"))
        gold_plan = ""
        
        prompt = build_hint_prompt(category, product_category, goal, gold_plan, row["total_reward"], keep_from_instruction(str(my_plan)))
        try:
            resp = agent.act(prompt)
        except Exception as e:
            tqdm.write(f"[WARN] model call failed for env_idx={env_idx}: {e}")
            pbar.update(1); continue

        parsed = parse_model_json(resp)
        if not parsed or "hints" not in parsed or not isinstance(parsed["hints"], list):
            tqdm.write(f"[WARN] model did not return valid JSON hints for env_idx={env_idx}")
            pbar.update(1); continue

        hints_clean = []
        for h in parsed["hints"][:4]:
            if not isinstance(h, dict): 
                continue
            t = str(h.get("category","")).strip().lower()
            txt_raw = str(h.get("text","")).strip()
            if not txt_raw:
                continue

            # Enforce the same truncation you already use; use this for comparisons too
            txt = txt_raw[:240]
            hint_cat = (t or category_norm)

            hints_clean.append({
                "category": hint_cat,
                "text": txt,
            })
            all_hint_texts[txt] += 1


        if not hints_clean:
            pbar.update(1); continue

        rec = {
            "env_idx": env_idx,
            "category": category,
            # "product_category": product_category,
            "goal": goal,
            # "gold_plan": gold_plan,
            "hints": hints_clean,
        }
        append_ndjson(args.out, rec)
        kept += 1
        pbar.set_postfix({"kept": kept, "skipped": skipped})  # skipped = deduped hints
        pbar.update(1)

    pbar.close()
    print(f"[SUMMARY] kept={kept} | skipped={skipped} (similar hints) | total_failures={len(df)}")

        
# ---------- CLI ----------
def _cli():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--csv", default="data/hint_corpus_train_nolimit_withsummary.csv", help="alfworld_scores.csv / alfworld_results.csv")
    ap.add_argument("--csv", default="game_logs/stateact-no-thoughts_train_all/", help="alfworld_scores.csv / alfworld_results.csv")
    # ap.add_argument("--csv", default="game_logs/react_train/", help="alfworld_scores.csv / alfworld_results.csv")
    # ap.add_argument("--csv", default="game_logs/act_train/", help="alfworld_scores.csv / alfworld_results.csv")
    # ap.add_argument("--out", default="data/hints/extracted_hints_react2.ndjson", help="Output NDJSON path")
    # ap.add_argument("--out", default="data/hints/extracted_hints_self3_react.ndjson", help="Output NDJSON path")
    ap.add_argument("--out", default="data/hints/extracted_hints_state2.ndjson", help="Output NDJSON path")

    # model flags
    ap.add_argument("--llm_type", default="GPTChat", help="GPTChat or VLLMChat")
    ap.add_argument("--model", default="gpt-4o", help="Model id, e.g., gpt-4o-mini or Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--force_model", action="store_true", default=False, help="Use proposed model even if not whitelisted")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--max_tokens", type=int, default=600, help="Max generation tokens")

    # local-only (ignored by GPTChat)
    ap.add_argument("--max_model_len", type=int, default=32768, help="Context length for local models")
    ap.add_argument("--quantization", type=int, default=0, help="Quantization flag for local models")
    ap.add_argument("--gpus", type=int, default=1, help="Tensor parallel size for local models")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # OpenAI keys (optional; env vars also work)
    ap.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY", ""), help="OpenAI API key")
    ap.add_argument("--openai_base_url", default=os.getenv("OPENAI_BASE_URL", ""), help="Custom OpenAI base URL (Azure, proxy, etc.)")
    ap.add_argument("--openai_org", default=os.getenv("OPENAI_ORG", ""), help="OpenAI organization id")

    args = ap.parse_args()
    main(args)


if __name__ == "__main__":
    _cli()