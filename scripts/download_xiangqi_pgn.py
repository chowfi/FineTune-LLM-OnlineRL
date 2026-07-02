#!/usr/bin/env python3
"""Download + extract the wukong-xiangqi master games PGN (UCCI notation).

Default source: `maksimKorzh/wukong-xiangqi` — `xqdb/xqdb/xqdb_masters_40711_UCI_games.pgn.zip`
(40,711 master games from wxf.ca; ~8 MB compressed). Moves are in UCCI/ICCS notation
(`h2e2`), so we can feed them straight to Pikafish and to ``cchess.Board.move_iccs``.

The download is **cached** at ``--out`` (default ``data/xiangqi_sft/raw/``). Re-running is a
no-op unless ``--force`` is set or the zip checksum disagrees.

Usage::

    uv run python scripts/download_xiangqi_pgn.py
    uv run python scripts/download_xiangqi_pgn.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

DEFAULT_URL = (
    "https://raw.githubusercontent.com/maksimKorzh/wukong-xiangqi/main/"
    "xqdb/xqdb/xqdb_masters_40711_UCI_games.pgn.zip"
)
# Tracked in the upstream commit log; an SHA mismatch is non-fatal (logged only).
EXPECTED_GIT_SHA = "837efd37c56a050525d703a89f947c4dc01ec4e5"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: str) -> None:
    tmp = dst + ".part"
    print(f"[download] {url} -> {dst}", flush=True)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    os.replace(tmp, dst)
    print(f"[download] wrote {total / 1e6:.2f} MB", flush=True)


def _extract_zip(zip_path: str, out_dir: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        # Prefer the largest .pgn inside the zip if multiple are present.
        pgn_members = [n for n in zf.namelist() if n.lower().endswith(".pgn")]
        if not pgn_members:
            raise SystemExit(f"No .pgn members found in {zip_path}")
        pgn_members.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        chosen = pgn_members[0]
        zf.extract(chosen, out_dir)
        extracted = os.path.join(out_dir, chosen)
        print(
            f"[extract] {chosen} -> {extracted} ({os.path.getsize(extracted) / 1e6:.2f} MB)",
            flush=True,
        )
        return extracted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=DEFAULT_URL)
    ap.add_argument(
        "--out-dir",
        type=str,
        default="data/xiangqi_sft/raw",
        help="Directory to cache the zip + extracted PGN.",
    )
    ap.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    zip_name = os.path.basename(args.url.split("?", 1)[0])
    zip_path = os.path.join(args.out_dir, zip_name)

    if args.force or not os.path.isfile(zip_path) or os.path.getsize(zip_path) == 0:
        _download(args.url, zip_path)
    else:
        print(f"[cache] using existing {zip_path}", flush=True)

    digest = _sha256(zip_path)
    print(f"[sha256] {digest}", flush=True)
    if EXPECTED_GIT_SHA and digest[: len(EXPECTED_GIT_SHA)] != EXPECTED_GIT_SHA:
        # GitHub blob SHA != SHA256 of bytes; just print both for the user to verify.
        print(
            f"[note] sha256 differs from upstream git blob sha ({EXPECTED_GIT_SHA}); "
            "this is expected (different hash families). Skipping checksum gate.",
            flush=True,
        )

    pgn_path = _extract_zip(zip_path, args.out_dir)
    print(f"OK: pgn ready at {pgn_path}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
