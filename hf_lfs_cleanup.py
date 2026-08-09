"""Delete orphaned LFS objects from the HF Space to stay under the 1 GB cap.

The Space is deployed as a force-pushed single-commit repo, so every deploy
orphans the previous deploy's LFS blobs (parquet/pkl). HF counts those
unreachable blobs against the 1 GB storage limit until they are permanently
deleted. Run after a successful deploy (CI) or manually.

Uses HF_TOKEN from the environment. Only deletes objects NOT referenced by
the current main revision.
"""
import os
import sys

from huggingface_hub import HfApi

REPO = "ACP-Analytics/ACP-Dashboard"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; skipping LFS cleanup.")
        return 0
    api = HfApi(token=token)

    tree = api.list_repo_tree(REPO, repo_type="space", recursive=True, expand=True)
    referenced = {f.lfs.sha256 for f in tree if getattr(f, "lfs", None)}
    all_lfs = list(api.list_lfs_files(REPO, repo_type="space"))
    orphans = [f for f in all_lfs if f.file_oid not in referenced]

    gb = lambda objs: sum(int(f.size) for f in objs) / 1e9
    print(f"LFS: {len(all_lfs)} objects ({gb(all_lfs):.2f} GB); "
          f"referenced {len(all_lfs) - len(orphans)}, orphaned {len(orphans)} ({gb(orphans):.2f} GB)")
    if not orphans:
        return 0

    api.permanently_delete_lfs_files(REPO, orphans, repo_type="space")
    remaining = list(api.list_lfs_files(REPO, repo_type="space"))
    print(f"After cleanup: {len(remaining)} objects ({gb(remaining):.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
