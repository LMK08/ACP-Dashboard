"""Add the full V metric set to every role's weights + expose new Values in
metric_categories.

Outfield roles get: Corner Value, Free Kick Value, Throw-In Value,
                    Fouling Value, Other Value, Total Value (at weight 1.0)
GK roles get:       Total Value (at weight 1.0)

Existing weights are left untouched.

Backs up to config.yaml.full-v-pre.bak before writing.
"""
import shutil
from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

HERE = Path(__file__).parent
CFG = HERE / "config.yaml"
BAK = HERE / "config.yaml.full-v-pre.bak"

OUTFIELD_V_NEW = [
    "Corner Value", "Free Kick Value", "Throw-In Value",
    "Fouling Value", "Other Value", "Total Value",
]
GK_V_NEW = ["Total Value"]
GK_ROLES = {"Shot Stopper", "Cross Claimer", "Ball-playing GK"}

# metric_categories additions — where each new V metric fits
CATEGORY_ADDITIONS = {
    "output":    ["Total Value"],
    "passing":   ["Corner Value", "Free Kick Value", "Throw-In Value"],
    "defensive": ["Fouling Value", "Other Value"],
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

    # ---- 1. Add to role weights ----
    added_weight = 0
    for role, w in cfg["weights"].items():
        new_keys = GK_V_NEW if role in GK_ROLES else OUTFIELD_V_NEW
        for k in new_keys:
            if k not in w:
                w[k] = 1.0
                added_weight += 1
    print(f"Added {added_weight} V weight keys across {len(cfg['weights'])} roles")

    # ---- 2. Add to metric_categories ----
    added_cat = 0
    for cat_name, new_metrics in CATEGORY_ADDITIONS.items():
        existing = cfg["metric_categories"].get(cat_name, [])
        for m in new_metrics:
            if m not in existing:
                existing.append(m)
                added_cat += 1
        cfg["metric_categories"][cat_name] = existing
    print(f"Added {added_cat} metrics to metric_categories")

    with open(CFG, "w") as f:
        yaml.dump(cfg, f)
    print(f"\nWrote {CFG.name}")


if __name__ == "__main__":
    main()
