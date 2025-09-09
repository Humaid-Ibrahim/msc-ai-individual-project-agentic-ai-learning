import json
from difflib import SequenceMatcher
import argparse

def deduplicate_hints(data, fuzzy=False, threshold=0.85):
    """
    Deduplicate hints within each category.
    If fuzzy=True, uses difflib SequenceMatcher to merge near-duplicates.
    """
    total_removed = 0
    deduped = {}
    for cat, content in data.items():
        removed = 0
        seen = []
        unique_hints = []
        
        for h in content["hints"]:
            norm = " ".join(h.lower().strip().split())  # normalize spaces + lowercase
            if not fuzzy:
                if norm not in seen:
                    seen.append(norm)
                    unique_hints.append(h)
            else:
                # check similarity with existing
                if any(SequenceMatcher(None, norm, s).ratio() > threshold for s in seen):
                    removed += 1
                    continue
                seen.append(norm)
                unique_hints.append(h)
        
        deduped[cat] = {
            "total_unique_hints": len(unique_hints),
            "total_mentions": len(content["hints"]),
            "removed_hints": removed,
            "hints": unique_hints,
        }
        total_removed += removed
    print("total removed hints: ", total_removed)
    return deduped

# # Example usage
# with open("data/hints_by_type_self_generated3.json") as f:
#     data = json.load(f)

# # exact deduplication
# deduped = deduplicate_hints(data, fuzzy=True)
# with open("data/hints_deduped_self_generated3.json", "w") as f:
#     json.dump(deduped, f, indent=2)


# with open("data/hints/hints_by_type_self_state2.json") as f:
#     data = json.load(f)

# # exact deduplication
# deduped = deduplicate_hints(data, fuzzy=True)
# with open("data/hints/hints_deduped_self_state2.json", "w") as f:
#     json.dump(deduped, f, indent=2)


# with open("data/hints/hints_by_type_self_react2.json") as f:
#     data = json.load(f)

# # exact deduplication
# deduped = deduplicate_hints(data, fuzzy=True)
# with open("data/hints/hints_deduped_self_react2.json", "w") as f:
#     json.dump(deduped, f, indent=2)


# with open("data/hints/hints_by_type_self_act.json") as f:
#     data = json.load(f)

# # exact deduplication
# deduped = deduplicate_hints(data, fuzzy=True)
# with open("data/hints/hints_deduped_self_act.json", "w") as f:
#     json.dump(deduped, f, indent=2)




def main(args):
    with open(args.inp) as f:
        data = json.load(f)
    # exact deduplication
    deduped = deduplicate_hints(data, fuzzy=True)
    with open(args.out, "w") as f:
        json.dump(deduped, f, indent=2)


# ---------- CLI ----------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/hints/hints_by_type_self_state2.json", help="alfworld_scores.csv / alfworld_results.csv")
    ap.add_argument("--out", default="data/hints/hints_deduped_self_state2.json", help="Output NDJSON path")
    args = ap.parse_args()
    main(args)


if __name__ == "__main__":
    _cli()


