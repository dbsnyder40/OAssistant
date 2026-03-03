# manifest.py
"""
Code to build and validate manifest files for vectorstore rebuild detection.

build_file_manifest(folder: Path):
needs_rebuild(config):
"""

from pathlib import Path
# from datetime import datetime
import json


# Build file manifest to dectect changes

def build_file_manifest(folder: Path) -> dict:

    """
    Create a snapshot of all files in a folder (recursively).
    Used to detect changes that require vectorstore rebuild.
    """
 
    manifest = {}

    for path in folder.glob("**/*"):
        if path.is_file():
            manifest[str(path)] = {
                "mtime": path.stat().st_mtime,
                "size": path.stat().st_size,
            }

    return manifest
    
def needs_rebuild(config:dict) -> bool:

    """
    Determine whether the vectorstore needs rebuilding.
    Rebuild is triggered if:
      - No manifest exists
      - Manifest is unreadable
      - Corpus name changed
      - Files added/removed/modified
    """

    db_dir = Path(config["paths"]["vectorstore_dir"])
    papers_dir = Path(config["paths"]["papers_dir"])
    manifest_path = db_dir / "manifest.json"

    # No manifest → rebuild
    if not manifest_path.exists():
        return True

    try:
        with open(manifest_path, "r") as f:
            saved_manifest = json.load(f)
    except Exception:
        return True  # corrupted manifest → rebuild

    # Check corpus change
    current_corpus = config.get("corpus", "research")
    if saved_manifest.get("corpus") != current_corpus:
        return True

    # Check file changes
    current_files = build_file_manifest(papers_dir)
    if saved_manifest.get("files") != current_files:
        return True

    return False
