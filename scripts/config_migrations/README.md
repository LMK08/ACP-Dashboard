# config.yaml migrations (one-shot, already applied)

Run-once scripts from 2026-05-13 that introduced the GPA "Value" metric family
into `config.yaml` (weights, metric categories, distribution metrics) and then
trimmed and reordered it. Each one backs up `config.yaml` before writing and
needs `ruamel.yaml`, which is not in `requirements.txt`.

They are kept for provenance only. `config.yaml` has been hand-edited since
(see its git history); re-running any of these would clobber those edits.
Order they were applied in:

1. `_patch_v_weights.py` — add V-metric keys to every role's weights
2. `_rename_and_expose_v_metrics.py` — `_per_90` keys → `Value`, expose in categories
3. `_add_full_v_set.py` — full V set on every role
4. `_trim_to_core_v.py` — back to the core Shooting/Passing/Receiving/Dribbling/Interrupting/Total set
5. `_reorder_weights_by_category.py` — cluster weights by radar category
6. `_apply_user_weights.py` — replace the weights block with Lucas's hand-tuned values
