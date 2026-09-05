"""Rename V weight keys (_per_90 → Value), expose them in metric_categories,
and add them to distribution_metrics_by_position per role.

Backs up to config.yaml.v-rename-pre.bak before writing.
"""
import shutil
from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

HERE = Path(__file__).parent
CFG = HERE / "config.yaml"
BAK = HERE / "config.yaml.v-rename-pre.bak"

# Map old weight key → new display name (9 core V metrics)
RENAME = {
    "Shooting_per_90":        "Shooting Value",
    "Passing_per_90":         "Passing Value",
    "Receiving_per_90":       "Receiving Value",
    "Dribbling_per_90":       "Dribbling Value",
    "SetPiece_per_90":        "Set Piece Value",
    "Interrupting_per_90":    "Interrupting Value",
    "GK_Shotstopping_per_90": "Shot-Stopping Value",
    "GK_Handling_per_90":     "Handling Value",
    "GK_Sweeping_per_90":     "Sweeping Value",
}

# Which metric_categories group each V metric belongs in
CATEGORY_ADDITIONS = {
    "output":      ["Shooting Value"],
    "passing":     ["Passing Value", "Receiving Value", "Set Piece Value"],
    "defensive":   ["Interrupting Value"],
    "dribbling":   ["Dribbling Value"],
    "goalkeeping": ["Shot-Stopping Value", "Handling Value", "Sweeping Value"],
}

# Per-role additions to distribution_metrics_by_position — curated to the
# V metrics most relevant for that archetype (3–4 per role).
DISTRIBUTION_ADDITIONS = {
    "Shot Stopper":             ["Shot-Stopping Value", "Handling Value"],
    "Cross Claimer":            ["Sweeping Value", "Handling Value"],
    "Ball-playing GK":          ["Passing Value", "Sweeping Value"],
    "Ball-Playing Centerback":  ["Passing Value", "Interrupting Value", "Dribbling Value"],
    "Stopper":                  ["Interrupting Value"],
    "Athletic Centerback":      ["Interrupting Value", "Passing Value"],
    "Box-to-Box":               ["Passing Value", "Interrupting Value", "Dribbling Value"],
    "Holding Mid":              ["Passing Value", "Interrupting Value"],
    "Ball-Winning Mid":         ["Interrupting Value", "Passing Value"],
    "Deep-lying Playmaker":     ["Passing Value", "Receiving Value"],
    "Advanced Playmaker":       ["Passing Value", "Receiving Value", "Dribbling Value"],
    "Full Back":                ["Passing Value", "Interrupting Value"],
    "Wingback":                 ["Dribbling Value", "Passing Value", "Set Piece Value"],
    "Inverted Full Back":       ["Passing Value", "Interrupting Value"],
    "Wide Winger":              ["Receiving Value", "Dribbling Value", "Set Piece Value"],
    "Creative Winger":          ["Passing Value", "Receiving Value", "Dribbling Value"],
    "Inside Forward":           ["Shooting Value", "Receiving Value", "Dribbling Value"],
    "Shadow Striker":           ["Shooting Value", "Receiving Value"],
    "Mobile Striker":           ["Shooting Value", "Receiving Value", "Dribbling Value"],
    "Poacher":                  ["Shooting Value", "Receiving Value"],
    "Target Man":               ["Receiving Value", "Shooting Value"],
    "Pressing Forward":         ["Shooting Value", "Interrupting Value"],
}


def main() -> None:
    if not BAK.exists():
        shutil.copy(CFG, BAK)
        print(f"Backed up {CFG.name} → {BAK.name}")

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 200

    with open(CFG) as f:
        cfg = yaml.load(f)

    # ---- 1. Rename weights keys ----
    renamed_count = 0
    for role, w in cfg["weights"].items():
        new_w = CommentedMap()
        for k, v in w.items():
            if k in RENAME:
                new_w[RENAME[k]] = v
                renamed_count += 1
            else:
                new_w[k] = v
        cfg["weights"][role] = new_w
    print(f"Renamed {renamed_count} weight keys across {len(cfg['weights'])} roles")

    # ---- 2. Add to metric_categories ----
    added_cats = 0
    for cat_name, new_metrics in CATEGORY_ADDITIONS.items():
        existing = cfg["metric_categories"].get(cat_name, [])
        for m in new_metrics:
            if m not in existing:
                existing.append(m)
                added_cats += 1
        cfg["metric_categories"][cat_name] = existing
    print(f"Added {added_cats} metrics to metric_categories")

    # ---- 3. Add to distribution_metrics_by_position ----
    added_dist = 0
    for role, new_metrics in DISTRIBUTION_ADDITIONS.items():
        existing = cfg["distribution_metrics_by_position"].get(role, [])
        for m in new_metrics:
            if m not in existing:
                existing.append(m)
                added_dist += 1
        cfg["distribution_metrics_by_position"][role] = existing
    print(f"Added {added_dist} metrics to distribution_metrics_by_position")

    with open(CFG, "w") as f:
        yaml.dump(cfg, f)
    print(f"\nWrote {CFG.name}")


if __name__ == "__main__":
    main()
