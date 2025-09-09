#!/usr/bin/env python
# aggregate_hints_by_type.py
import argparse, json, pathlib, re
from collections import defaultdict, Counter


def title_type(t: str) -> str:
    return t.replace("-", " ").title()

def aggregate(in_path: str,
              top_k_triggers: int = 3,
              top_k_fixes: int = 3,
              max_env_types: int = 5):
    # type -> Counter(text -> count)
    counts = defaultdict(Counter)
    triggers = defaultdict(lambda: defaultdict(Counter))   # [type][text] -> Counter(trigger)
    fixes = defaultdict(lambda: defaultdict(Counter))      # [type][text] -> Counter(fix)
    envtypes = defaultdict(lambda: defaultdict(Counter))   # [type][text] -> Counter(env_type)
    # raise ValueError
                  
    n_lines = 0
    with open(in_path) as f:
        for line in f:
            n_lines += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            env_type = str(rec.get("category","unknown"))
            goal = str(rec.get("goal","unknown"))
            # print(env_type)
            
            for h in rec.get("hints", []) or []:
                # t = normalize_type(h.get("type",""))
                t = env_type
                # print(t)
                txt = (h.get("text") or "").strip()
                if not txt:
                    continue
                counts[t][txt] += 1
                # trig = (h.get("when_to_trigger") or "").strip()
                # fix  = (h.get("example_fix") or "").strip()
                # if trig:
                #     triggers[t][txt][trig] += 1
                # if fix:
                #     fixes[t][txt][fix] += 1
                if env_type:
                    envtypes[t][txt][env_type] += 1
    # print(envtypes)
    # Build output
    sections = {}
    # types = ["beauty", "food", "fashion", "furniture", "electronics", "grocery", "garden"]
    types = ["beauty", "food", "fashion", "furniture", "electronics"]
    for t in types:
        print(t)
        items = []
        for text, c in counts[t].most_common():
            # top_trigs = [k for k,_ in triggers[t][text].most_common(top_k_triggers)]
            # top_fix   = [k for k,_ in fixes[t][text].most_common(top_k_fixes)]
            top_envs  = dict(envtypes[t][text].most_common(max_env_types))
            # items.append({
            #     "text": text,
            #     "count": c,
            #     # "top_triggers": top_trigs,
            #     # "top_example_fixes": top_fix,
            #     # "env_types": top_envs,
            # })
            # print(text)
            items.append(text)
            
        # sub_section = {}
        # sub_section[title_type(t)] = {}
        sub_section = {
            "total_unique_hints": len(items),
            "total_mentions": sum(counts[t].values()),
            "hints": items
        }
        # if t == "garden":
        #     t = "furniture"
        # if t == "grocery":
        #     t = "food"
            
        # sections[title_type(t).lower()] = {}
        sections[title_type(t).lower()] = sub_section
        # sections.append(sub_section)
        # sections.append({
        #     "type": title_type(t),
        #     "total_unique_hints": len(items),
        #     "total_mentions": sum(counts[t].values()),
        #     "hints": items
        # })
    return sections



# inp = "data/extracted_hints_self_generated3.ndjson"
# out = "data/hints_by_type_self_generated3.json"

# inp = "data/hints/extracted_hints_act.ndjson"
# out = "data/hints/hints_by_type_self_act.json"

# inp = "data/hints/extracted_hints_react2.ndjson"
# out = "data/hints/hints_by_type_self_react2.json"

# inp = "data/hints/extracted_hints_state2.ndjson"
# out = "data/hints/hints_by_type_self_state2.json"
# report = aggregate(inp, 3, 3, 5)
# pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
# with open(out, "w") as f:
#     json.dump(report, f, indent=2)
# print(f"[OK] wrote → {out}")

def main(args):
    report = aggregate(args.inp, 3, 3, 5)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] wrote → {args.out}")

# ---------- CLI ----------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/hints/extracted_hints_state2.ndjson", help="alfworld_scores.csv / alfworld_results.csv")
    ap.add_argument("--out", default="data/hints/hints_by_type_self_state2.json", help="Output NDJSON path")
    args = ap.parse_args()
    main(args)


if __name__ == "__main__":
    _cli()


