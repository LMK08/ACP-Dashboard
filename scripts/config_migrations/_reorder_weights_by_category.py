"""Reorder each role's weights in config.yaml so metrics cluster by category
(same color on the radar). V metrics land inside their category block next
to same-type classical metrics.

Category assignment is taken from config.yaml's `metric_categories` section.
Metrics not assigned to any category land in an 'other' bucket at the end
(preserving their original relative order).

Backs up to config.yaml.reorder-pre.bak before writing.
"""
import shutil
from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

HERE = Path(__file__).parent
CFG = HERE / "config.yaml"
BAK = HERE / "config.yaml.reorder-pre.bak"

# Category blocks in the order we want them to appear on the radar
# (roughly attacking → midfield → defending → goalkeeping)
CATEGORY_ORDER = ["output", "passing", "dribbling", "defensive", "goalkeeping"]


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

    # Build lookup: metric_name → category
    # Precedence matches the radar's color-assignment order in app.py
    # (plot_comparison_radar / create_radar_with_distributions):
    # output > passing > defensive > dribbling > goalkeeping
    # A metric present in multiple categories takes the FIRST match so the
    # bucket and the radar color agree.
    CATEGORY_PRECEDENCE = ["output", "passing", "defensive", "dribbling", "goalkeeping"]
    metric_to_cat: dict[str, str] = {}
    for cat in CATEGORY_PRECEDENCE:
        for m in cfg["metric_categories"].get(cat, []):
            if m not in metric_to_cat:
                metric_to_cat[m] = cat

    reordered_count = 0
    for role, weights in cfg["weights"].items():
        # Bucket metrics by category, preserving original order within bucket
        buckets: dict[str, list[tuple[str, object]]] = {
            cat: [] for cat in CATEGORY_ORDER
        }
        buckets["other"] = []
        for m, w in weights.items():
            cat = metric_to_cat.get(m, "other")
            if cat not in buckets:
                cat = "other"
            buckets[cat].append((m, w))

        # Rebuild weights in category-block order
        new_weights = CommentedMap()
        for cat in CATEGORY_ORDER + ["other"]:
            for m, w in buckets[cat]:
                new_weights[m] = w

        # Sanity: same set of keys
        assert set(new_weights.keys()) == set(weights.keys()), \
            f"key mismatch in role {role}"

        cfg["weights"][role] = new_weights
        reordered_count += 1

    with open(CFG, "w") as f:
        yaml.dump(cfg, f)
    print(f"Reordered {reordered_count} roles. Categories cluster in this order: "
          f"{' → '.join(CATEGORY_ORDER)} → other")


if __name__ == "__main__":
    main()
