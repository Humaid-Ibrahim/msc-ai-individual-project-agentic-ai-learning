#!/usr/bin/env python
# aggregate_hints_by_type.py
import argparse, json, pathlib, re
from collections import defaultdict, Counter


def title_type(t: str) -> str:
    return t.replace("-", " ").title()

def aggregate(in_path: str,
              top_k_triggers: int = 3,
              top_k_fixes: int = 3,
              max_env_types: int = 6):
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

            env_type = str(rec.get("env_type","unknown"))
            # print(env_type)
            for h in rec.get("hints", []) or []:
                # t = normalize_type(h.get("type",""))
                t = env_type
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

    # Build output
    sections = {}
    # types_sorted = [t for t in CANON_ORDER if t in counts] + sorted(
        # [t for t in counts.keys() if t not in CANON_ORDER]
    # )
    types_sorted = ["clean", "cool", "examine", "heat", "put", "puttwo"]
    for t in types_sorted:
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
            items.append(text)
            
        sub_section = {}
        # sub_section[title_type(t)] = {}
        sub_section = {
            "total_unique_hints": len(items),
            "total_mentions": sum(counts[t].values()),
            "hints": items
        }
        sections[title_type(t).lower()] = {}
        sections[title_type(t).lower()] = sub_section
        # sections.append(sub_section)
        # sections.append({
        #     "type": title_type(t),
        #     "total_unique_hints": len(items),
        #     "total_mentions": sum(counts[t].values()),
        #     "hints": items
        # })
    return sections
    # return {
    #     # "summary": {
    #     #     "files_processed": 1,
    #     #     "lines_read": n_lines,
    #     #     "total_types": len(sections),
    #     #     "total_unique_hints": sum(len(s["hints"]) for s in sections),
    #     #     "total_mentions": sum(s["total_mentions"] for s in sections),
    #     # },
    #     "by_type": sections
    # }


inp = "data/hints_selfonly2_react.ndjson"
out = "data/hints_selfonly_clean2_react.ndjson"
report = aggregate(inp, 3, 3, 6)
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(report, f, indent=2)
print(f"[OK] wrote → {out}")

