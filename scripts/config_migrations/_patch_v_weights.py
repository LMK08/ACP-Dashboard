"""Add GPA V-metric keys to every role's weights block in config.yaml.

Outfield roles get the 6 outfield V metrics; GK roles get the 4 GK V metrics.
Weights use my role-informed best-guesses where I had them; all other V metrics
for a role are inserted with a neutral default of 1.0 so the user can tune.

Run once:
    python _patch_v_weights.py

Backs up the original to config.yaml.bak first.
"""
import shutil
from pathlib import Path
from ruamel.yaml import YAML

HERE = Path(__file__).parent
CFG = HERE / "config.yaml"
BAK = HERE / "config.yaml.bak"

OUTFIELD_V = [
    "Shooting_per_90", "Passing_per_90", "Receiving_per_90",
    "Dribbling_per_90", "SetPiece_per_90", "Interrupting_per_90",
]
GK_V = [
    "GK_Shotstopping_per_90", "GK_Handling_per_90",
    "GK_Sweeping_per_90", "Passing_per_90",
]
GK_ROLES = {"Shot Stopper", "Cross Claimer", "Ball-playing GK"}

# Best-guess weights per role; anything not listed defaults to 1.0 for tuning.
ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "Shot Stopper":             {"GK_Shotstopping_per_90": 12.0, "GK_Handling_per_90": 5.0},
    "Cross Claimer":            {"GK_Sweeping_per_90": 15.0, "GK_Handling_per_90": 6.0},
    "Ball-playing GK":          {"Passing_per_90": 10.0, "GK_Sweeping_per_90": 4.0},
    "Mobile Striker":           {"Shooting_per_90": 8.0, "Receiving_per_90": 6.0, "Dribbling_per_90": 6.0},
    "Shadow Striker":           {"Shooting_per_90": 10.0, "Receiving_per_90": 10.0, "Dribbling_per_90": 5.0},
    "Poacher":                  {"Shooting_per_90": 15.0, "Receiving_per_90": 4.0},
    "Target Man":               {"Receiving_per_90": 10.0, "Shooting_per_90": 6.0, "Interrupting_per_90": 3.0},
    "Pressing Forward":         {"Shooting_per_90": 6.0, "Interrupting_per_90": 6.0},
    "Box-to-Box":               {"Passing_per_90": 6.0, "Receiving_per_90": 5.0, "Dribbling_per_90": 5.0, "Interrupting_per_90": 6.0, "Shooting_per_90": 4.0},
    "Ball-Winning Mid":         {"Interrupting_per_90": 12.0, "Passing_per_90": 4.0},
    "Holding Mid":              {"Passing_per_90": 8.0, "Interrupting_per_90": 8.0},
    "Deep-lying Playmaker":     {"Passing_per_90": 15.0, "Receiving_per_90": 5.0, "Interrupting_per_90": 4.0},
    "Advanced Playmaker":       {"Passing_per_90": 12.0, "Receiving_per_90": 10.0, "Dribbling_per_90": 6.0, "SetPiece_per_90": 3.0},
    "Wide Winger":              {"Passing_per_90": 6.0, "Receiving_per_90": 6.0, "Dribbling_per_90": 8.0, "SetPiece_per_90": 5.0},
    "Creative Winger":          {"Passing_per_90": 10.0, "Receiving_per_90": 10.0, "Dribbling_per_90": 8.0},
    "Inside Forward":           {"Shooting_per_90": 10.0, "Receiving_per_90": 8.0, "Dribbling_per_90": 6.0},
    "Full Back":                {"Passing_per_90": 6.0, "Interrupting_per_90": 6.0, "Dribbling_per_90": 3.0},
    "Wingback":                 {"Passing_per_90": 5.0, "Receiving_per_90": 4.0, "Dribbling_per_90": 6.0, "SetPiece_per_90": 3.0, "Interrupting_per_90": 5.0},
    "Inverted Full Back":       {"Passing_per_90": 10.0, "Interrupting_per_90": 5.0},
    "Ball-Playing Centerback":  {"Passing_per_90": 15.0, "Interrupting_per_90": 6.0},
    "Stopper":                  {"Interrupting_per_90": 12.0, "Passing_per_90": 3.0},
    "Athletic Centerback":      {"Interrupting_per_90": 8.0, "Passing_per_90": 6.0},
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

    weights = cfg["weights"]
    added_total = 0
    for role, role_weights in weights.items():
        v_keys = GK_V if role in GK_ROLES else OUTFIELD_V
        picks = ROLE_WEIGHTS.get(role, {})
        added_here = 0
        for k in v_keys:
            if k in role_weights:
                continue  # don't clobber if already present
            role_weights[k] = picks.get(k, 1.0)
            added_here += 1
        added_total += added_here
        print(f"  {role}: +{added_here} V metrics (total keys now {len(role_weights)})")

    with open(CFG, "w") as f:
        yaml.dump(cfg, f)
    print(f"\nWrote {CFG.name} with {added_total} V keys added across {len(weights)} roles")


if __name__ == "__main__":
    main()
