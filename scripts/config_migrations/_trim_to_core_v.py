"""Trim V metrics back to the core set.

Keeps:
  - Outfield roles: Shooting, Passing, Receiving, Dribbling, Interrupting, Total
  - GK roles:       Shot-Stopping, Handling, Sweeping, Passing, Total

Removes (from weights, metric_categories, distribution_metrics_by_position):
  Set Piece Value, Corner Value, Free Kick Value, Throw-In Value,
  Fouling Value, Other Value

Parquet columns are left intact (auxiliary data available if needed later).
"""
import shutil
from pathlib import Path
from ruamel.yaml import YAML

HERE = Path(__file__).parent
CFG = HERE / "config.yaml"
BAK = HERE / "config.yaml.trim-pre.bak"

REMOVE = {
    "Set Piece Value",
    "Corner Value",
    "Free Kick Value",
    "Throw-In Value",
    "Fouling Value",
    "Other Value",
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

    # ---- 1. Remove from role weights ----
    removed_w = 0
    for role, w in cfg["weights"].items():
        for k in list(w.keys()):
            if k in REMOVE:
                del w[k]
                removed_w += 1
    print(f"Removed {removed_w} weight entries from {len(cfg['weights'])} roles")

    # ---- 2. Remove from metric_categories ----
    removed_c = 0
    for cat_name, metrics in cfg["metric_categories"].items():
        kept = [m for m in metrics if m not in REMOVE]
        removed_c += len(metrics) - len(kept)
        cfg["metric_categories"][cat_name] = kept
    print(f"Removed {removed_c} entries from metric_categories")

    # ---- 3. Remove from distribution_metrics_by_position ----
    removed_d = 0
    for role, metrics in cfg["distribution_metrics_by_position"].items():
        kept = [m for m in metrics if m not in REMOVE]
        removed_d += len(metrics) - len(kept)
        cfg["distribution_metrics_by_position"][role] = kept
    print(f"Removed {removed_d} entries from distribution_metrics_by_position")

    with open(CFG, "w") as f:
        yaml.dump(cfg, f)
    print(f"\nWrote {CFG.name}")


if __name__ == "__main__":
    main()
