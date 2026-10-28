"""Download the complete public 22-sequence probe subset over HF Mirror, no proxy."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = "kingJulio/touch-and-go-probe"
REVISION = "ba0bee2d863aabee6b3d533a4e15618d3d149ab1"
LABEL_REVISION = "edd0e04430b966fcf08b8e17fe9955f71a2dcbed"


def curl(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    subprocess.run([
        "curl", "--noproxy", "*", "--fail", "--location", "--silent",
        "--show-error", "--retry", "4", "--retry-delay", "2",
        "--connect-timeout", "20", "--max-time", "600", url,
        "--output", str(temporary),
    ], check=True)
    temporary.replace(destination)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    source_checksums = json.loads((ROOT / "references/source_checksums.json").read_text())
    for filename in ['category_reference.txt', 'train_nointer_touch_instances.txt']:
        target = ROOT / 'references' / filename
        if not target.exists():
            curl(f'https://raw.githubusercontent.com/kaistmm/SeeingThroughTouch/{LABEL_REVISION}/datasets/touch_and_go/metadata/{filename}', target)
        if sha256(target) != source_checksums[filename]:
            raise RuntimeError(f"Metadata SHA256 mismatch: {target}")
    metadata = ROOT / "references/hf_dataset_tree.json"
    if not metadata.exists():
        curl(f"https://hf-mirror.com/api/datasets/{REPO}/tree/{REVISION}?recursive=true&limit=1000", metadata)
    entries = [x for x in json.loads(metadata.read_text()) if x["type"] == "file" and not x["path"].startswith(".")]

    def fetch(entry):
        path = ROOT / "data/raw/touch_and_go" / entry["path"]
        expected = entry.get("lfs", {}).get("oid")
        if not (path.exists() and path.stat().st_size == entry["size"] and (not expected or sha256(path) == expected)):
            curl(f"https://hf-mirror.com/datasets/{REPO}/resolve/{REVISION}/{entry['path']}?download=true", path)
        actual = sha256(path)
        if path.stat().st_size != entry["size"] or (expected and actual != expected):
            raise RuntimeError(f"Size/SHA256 mismatch: {path}")
        print(f"verified {entry['path']} ({entry['size']:,} bytes)", flush=True)
        return {"path": str(path.relative_to(ROOT)), "bytes": entry["size"], "sha256": actual}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(fetch, entries))
    manifest = {"repo": REPO, "revision": REVISION, "endpoint": "https://hf-mirror.com", "proxy": False,
                "scope": "complete public probe mirror; subset of the original Touch and Go dataset",
                "sequences": len({Path(x["path"]).parent.name for x in files}),
                "metadata_repo": "kaistmm/SeeingThroughTouch", "metadata_revision": LABEL_REVISION,
                "total_bytes": sum(x["bytes"] for x in files), "files": files}
    (ROOT / "data/download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
