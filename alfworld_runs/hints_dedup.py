import json
from difflib import SequenceMatcher

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

# Example usage
with open("data/hints_selfonly_clean2_react.ndjson") as f:
    data = json.load(f)

# exact deduplication
deduped = deduplicate_hints(data, fuzzy=True)
with open("data/hints_by_type_self_generated2_react.json", "w") as f:
    json.dump(deduped, f, indent=2)
