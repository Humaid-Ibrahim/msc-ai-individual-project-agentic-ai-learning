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

# # ----------------- OPTIONAL LOCAL MODEL -----------------
# try:
#     # If you still want local inference
#     from LLamp.llamp.llms.local import VLLMChat  # keep your path
# except Exception:
#     VLLMChat = None

# ----------------- GPT (OpenAI) MODEL -------------------
# Minimal wrapper around OpenAI's Chat Completions API
class GPTChat:
    def __init__(self,
                 model: str = "gpt-4o-mini",
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

NAV_ACTIONS = {
    "MoveAhead","RotateLeft","RotateRight","LookUp","LookDown",
    "MoveBack","MoveRight","MoveLeft"
}

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

# ---------- CSV helpers ----------
TRACE_COL_CANDIDATES = ["trace_file","trace_path","trace","trajectory_path","trace_json"]
PDDL_COL_CANDIDATES  = ["env_reference","env_referencem","game_folder","pddl_dir","traj_path","traj_data"]

def _read_main_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need_any_trace = any(c in df.columns for c in TRACE_COL_CANDIDATES)
    need_any_pddl  = any(c in df.columns for c in PDDL_COL_CANDIDATES) or "env_type" in df.columns
    if not need_any_trace:
        raise ValueError(f"CSV must contain one of {TRACE_COL_CANDIDATES}")
    if not need_any_pddl:
        raise ValueError(f"CSV must contain one of {PDDL_COL_CANDIDATES} or at least env_type")
    for must in ["success","env_idx"]:
        if must not in df.columns:
            raise ValueError(f"CSV missing required column '{must}'")
    if "env_type" not in df.columns:
        df["env_type"] = "unknown"
    return df

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---------- File loading ----------
def _load_chat(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

# ---------- Resolver for traj_data.json ----------
def resolve_traj(env_ref: str, data_root: pathlib.Path, splits: List[str]) -> Optional[pathlib.Path]:
    p = data_root / env_ref
    if p.is_dir() and (p / "traj_data.json").is_file():
        return p / "traj_data.json"
    if p.name == "traj_data.json" and p.is_file():
        return p
    if p.name == "game.tw-pddl" and p.with_name("traj_data.json").is_file():
        return p.with_name("traj_data.json")
    for sp in splits:
        base = data_root / sp / env_ref
        if base.is_dir() and (base / "traj_data.json").is_file():
            return base / "traj_data.json"
        if base.name == "traj_data.json" and base.is_file():
            return base
        if base.name == "game.tw-pddl" and base.with_name("traj_data.json").is_file():
            return base.with_name("traj_data.json")
    return None

def find_traj_from_row(row: pd.Series,
                       data_root: pathlib.Path,
                       splits: List[str]) -> Optional[pathlib.Path]:
    for c in ["traj_path","traj_data"]:
        if c in row and isinstance(row[c], str) and row[c].endswith("traj_data.json"):
            p = pathlib.Path(row[c])
            if p.is_file():
                return p
    for c in ["game_folder","pddl_dir"]:
        if c in row and isinstance(row[c], str) and row[c]:
            cand = pathlib.Path(row[c])
            if cand.is_dir() and (cand / "traj_data.json").is_file():
                return cand / "traj_data.json"
            if cand.name == "game.tw-pddl" and cand.with_name("traj_data.json").is_file():
                return cand.with_name("traj_data.json")
    for c in ["env_reference","env_referencem"]:
        if c in row and isinstance(row[c], str) and row[c]:
            t = resolve_traj(row[c], data_root, splits)
            if t:
                return t
    return None

# ---------- Trace parsing helpers ----------
def extract_turns_min(trace: List[Dict[str, Any]], max_pairs: int = 8) -> List[Tuple[str,str]]:
    pairs: List[Tuple[str,str]] = []
    last_action = None
    for msg in trace:
        role = msg.get("role","")
        content = str(msg.get("content",""))
        if role == "assistant":
            m = re.search(r"(?:^|\n)action\s*:\s*(.+)", content, re.I)
            if m:
                last_action = m.group(1).strip()
        elif role == "user":
            obs = content.strip()
            if last_action is not None:
                pairs.append((last_action, obs))
                last_action = None
    return pairs[-max_pairs:]

def summarize_failure(pairs: List[Tuple[str,str]]) -> str:
    lines = []
    for i, (a,o) in enumerate(pairs, 1):
        o_short = re.sub(r"\s+", " ", o).strip()
        if len(o_short) > 180:
            o_short = o_short[:177] + "…"
        lines.append(f"{i}. action={a} | obs={o_short}")
    return "\n".join(lines)

import re, json, pathlib
from typing import Any, Dict, List, Optional, Tuple

def _fmt_args(args):
    if args is None:
        return ""
    if isinstance(args, (list, tuple)):
        return ", ".join(map(str, args))
    if isinstance(args, dict):
        pref = ["object","object_target","objectId","receptacle","receptacle_target","receptacleObjectId","container","location"]
        ordered = [args[k] for k in pref if k in args]
        ordered += [v for k, v in args.items() if k not in pref]
        return ", ".join(map(str, ordered))
    return str(args)

TASK_RE = re.compile(r"Your task is to:\s*(.+)", re.I)
GOAL_RE = re.compile(r"^\s*>?goal:\s*(.+)$", re.I | re.M)

def goal_from_agent_trace(trace: List[Dict]) -> Optional[str]:
    after_task_flag = False
    for msg in trace:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))

        if role == "system" and "Here is the task" in content:
            after_task_flag = True
            continue

        if after_task_flag and role == "user":
            m = TASK_RE.search(content)
            if m:
                return m.group(1).strip().rstrip(".")
            for ln in content.splitlines():
                m2 = TASK_RE.search(ln)
                if m2:
                    return m2.group(1).strip().rstrip(".")
            after_task_flag = False

    for msg in trace:
        if msg.get("role") != "user":
            continue
        m = TASK_RE.search(str(msg.get("content", "")))
        if m:
            return m.group(1).strip().rstrip(".")
    blob = "\n".join(str(m.get("content","")) for m in trace)
    m = GOAL_RE.search(blob)
    if m:
        return m.group(1).strip().rstrip(".")
    return None

def gold_from_traj(traj_json: Dict[str, Any],
                   traj_path: Optional[pathlib.Path] = None
                   ) -> List[str]:
    plan = traj_json.get("plan") or {}
    hi_candidates = [
        plan.get("high_pddl"),
        plan.get("high_actions"),
        traj_json.get("high_pddl"),
        traj_json.get("high_actions"),
    ]
    hi = next((x for x in hi_candidates if isinstance(x, (list, tuple))), [])
    gold_plan: List[str] = []
    for step in hi:
        if isinstance(step, str):
            gold_plan.append(step.strip()); continue
        if not isinstance(step, dict):
            gold_plan.append(str(step)); continue
        if isinstance(step.get("discrete_action"), dict):
            da = step["discrete_action"]
            act = da.get("action") or "UNKNOWN"
            args = da.get("args")
            gold_plan.append(f"{act}({_fmt_args(args)})"); continue
        act = step.get("action") or step.get("op") or step.get("name") or step.get("predicate")
        if act:
            args = step.get("args") or step.get("arguments") or step.get("objects") or step.get("params")
            gold_plan.append(f"{act}({_fmt_args(args)})"); continue
        if isinstance(step.get("planner_action"), dict):
            pa = step["planner_action"]
            act = pa.get("action") or "UNKNOWN"
            if act == "GotoLocation":
                loc = pa.get("location"); gold_plan.append(f"{act}({loc if loc else ''})")
            elif act in ("PutObject","PickupObject","OpenObject","CloseObject","ToggleObjectOn","ToggleObjectOff","SliceObject","UseObject"):
                args = []
                oid = pa.get("objectId") or (pa.get("coordinateObjectId") or [None, None])[0]
                rid = pa.get("receptacleObjectId") or (pa.get("coordinateReceptacleObjectId") or [None, None])[0]
                if oid: args.append(oid)
                if rid: args.append(rid)
                gold_plan.append(f"{act}({_fmt_args(args)})")
            else:
                gold_plan.append(f"{act}()")
            continue
        gold_plan.append(json.dumps(step, sort_keys=True))
    return gold_plan

# ---------- Prompting ----------
HINT_TYPES = ["precondition","affordance","ordering","failure-pattern","best-practice","goal-clarification"]

def build_hint_prompt(env_type: str,
                      goal: Optional[str],
                      gold_plan: List[str],
                      failure_summary: str) -> str:
    goal_txt = goal or "(goal text unavailable)"
    plan_txt = "\n".join(f"- {s}" for s in gold_plan) if gold_plan else "(no gold plan found)"
# You are diagnosing why an ALFWorld agent failed and creating runtime HINTS to avoid future failures in similar tasks.
# Figure out what the agent did wrong and provide runtime HINTS to the agent when it starts another similar task.

    return f"""
You are diagnosing why a household agent failed and creating runtime HINTS to avoid future failures in similar tasks.

Environment type: {env_type}
Task goal: {goal_txt}

=======
Steps before failure (action → observation):
{failure_summary}
=======

Emit STRICT JSON with this schema:
{{
  "hints": [
    {{
      "env_type": \"{env_type}",
      "text": "≤120 chars, imperative advice the agent can follow in future for similar environment types"
    }}
  ]
}}

Rules:
- Focus on errors that explain THIS failure; provide hints to avoid failures on SIMILAR tasks.
- Make it generally applicable.
- Use placeholders like {{object}}, {{container}}, {{location}}, {{page}}, {{item}} instead of numbers/IDs.
- 1–4 high-value hints max. No duplicates. No meta commentary.
- JSON only. No extra text.
""".strip()
    
    # return f"""
# You are diagnosing why an ALFWorld agent failed and creating runtime HINTS to avoid future failures in similar tasks.

# Environment type: {env_type}
# Task goal: {goal_txt}

# =======
# Correct high-level PDDL plan (reference):
# {plan_txt}
# =======
# Last steps before failure (action → observation):
# {failure_summary}
# =======

# Emit STRICT JSON with this schema:
# {{
#   "hints": [
#     {{
#       "env_type": \"{env_type}",
#       "text": "≤120 chars, imperative advice the agent can follow in future for similar environment types"
#     }}
#   ]
# }}

# Rules:
# - Focus on errors that explain THIS failure; provide hints to avoid failures on SIMILAR tasks.
# - Make it generally applicable.
# - Use placeholders like {{object}}, {{container}}, {{location}}, {{page}}, {{item}} instead of numbers/IDs.
# - 1–4 high-value hints max. No duplicates. No meta commentary.
# - JSON only. No extra text.
# """.strip()

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

def strip_fewshot(conversation):
    """
    Remove few-shot examples in the system prompt and replace with
    'Interact with a household to solve a task.\n\nHere is the task.\n'
    """
    new_conv = []
    for msg in conversation:
        if msg["role"] == "system":
            # replace its content
            new_msg = msg.copy()
            new_msg["content"] = "Interact with a household to solve a task.\n\nHere is the task.\n"
            new_conv.append(new_msg)
        else:
            new_conv.append(msg)
    return new_conv


# ---------- Main pipeline ----------
def main(args) -> None:
    df = _read_main_csv(args.csv)
    trace_col = _pick_col(df, TRACE_COL_CANDIDATES)
    pddl_col  = _pick_col(df, PDDL_COL_CANDIDATES)

    df = df[df["success"] == False].reset_index(drop=True)
    if df.empty:
        print("[INFO] No failures in CSV — nothing to extract."); return

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

    data_root = pathlib.Path(args.data_root) if args.data_root else None
    splits = (args.splits.split(",") if args.splits else ["train","valid_seen","valid_unseen"])

    kept = 0; skipped = 0; all_hint_texts = Counter()
    pbar = tqdm(total=len(df), desc="Extracting hints", unit="env")

    for _, row in df.iterrows():

        # if _ < 30:
        #     continue
        
        env_idx  = int(row["env_idx"])
        env_type = str(row.get("env_type","unknown"))
        trace_path_raw = str(row.get(trace_col, "")) if trace_col else ""
        trace_path = trace_path_raw
        if trace_path and not os.path.isabs(trace_path) and args.base:
            trace_path = os.path.join(args.base, trace_path)

        key = (env_idx, trace_path_raw)
        if key in already:
            skipped += 1; pbar.update(1); continue

        try:
            trace = _load_chat(trace_path)
        except Exception as e:
            tqdm.write(f"[WARN] cannot load trace for env_idx={env_idx}: {e}")
            pbar.update(1); continue
        trace = strip_fewshot(trace)
        # print(trace)

        
        traj_path = None
        if pddl_col:
            traj_path = find_traj_from_row(row, data_root or pathlib.Path("."), splits)
        if traj_path is None and data_root is not None:
            pass

        traj_json = _read_json(str(traj_path)) if traj_path else None
        gold_plan = gold_from_traj(traj_json, pathlib.Path(traj_path)) if traj_json else []
        goal = goal_from_agent_trace(trace)

        # pairs = extract_turns_min(trace, max_pairs=args.max_pairs)
        # failure_summary = summarize_failure(pairs) if pairs else "(no pairs extracted)"
        
        # print(failure_summary)
        # print("*"*20)
        # print(gold_plan)
        # raise ValueError
        
        prompt = build_hint_prompt(env_type, goal, gold_plan, trace)
        # print(prompt)
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
            t = str(h.get("env_type","")).strip().lower()
            txt = str(h.get("text","")).strip()
            if not txt:
                continue
            hints_clean.append({
                "env_type": t or env_type,
                "text": txt[:240],
            })
            all_hint_texts[txt] += 1

        if not hints_clean:
            pbar.update(1); continue

        rec = {
            "env_idx": env_idx,
            "env_type": env_type,
            "trace_file": trace_path_raw,
            "traj_path": str(traj_path) if traj_path else None,
            "goal": goal,
            "gold_plan": gold_plan,
            "hints": hints_clean,
        }
        append_ndjson(args.out, rec)
        kept += 1
        pbar.set_postfix({"kept": kept, "skipped": skipped})
        pbar.update(1)

    pbar.close()
    print(f"[SUMMARY] kept={kept} | skipped={skipped} | total_failures={len(df)}")
    
# ---------- CLI ----------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="alfworld_scores.csv / alfworld_results.csv")
    ap.add_argument("--base", default="", help="Base dir for relative trace paths")
    ap.add_argument("--data_root", default="", help="ALFWorld json_2.1.1 root")
    ap.add_argument("--splits", default="train,valid_seen,valid_unseen", help="Comma-separated splits to search")
    ap.add_argument("--out", default="hints.ndjson", help="Output NDJSON path")
    ap.add_argument("--max_pairs", type=int, default=99, help="Max (action, obs) pairs to summarize")

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

"""
python expel_insight_extraction_goldtruth-GPT.py --csv ""



 """