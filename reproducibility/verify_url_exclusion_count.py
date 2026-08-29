"""Full-scale acceptance test for the loader url exclusion.

Applies the exact keep-mask expression from dynamic_data_loader.load_source_slice
(url tail extract + is_in + fill_null(True)) to the full slim craft_curated
bundle and asserts:

- rows dropped == 77,469 (every must-exclude url is present exactly once; the
  slim build dedupes cjd by url)
- unique matched urls == 77,469 (no must-exclude url missing from the data)
- every noncjd row survives (the null-mask path keeps tail-less metadata)

Reads the local slim mirror at $SLIM_LOCAL_DIR after the same
freshness check as extract_slim_urls.py; pass --use-gcs to scan GCS directly.
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("POLARS_MAX_THREADS", str(os.cpu_count()))
import polars as pl

LOCAL_DIR = os.environ.get("SLIM_LOCAL_DIR", "craft_curated_slim")
GCS_PREFIX = os.environ.get("GCS_PREFIX", "gs://YOUR_BUCKET/multilingual_data")
GCS_GLOB = os.environ.get("SLIM_GLOB", f"{GCS_PREFIX}/craft_curated_slim_*.parquet")
URLS_FILE = os.environ.get("MUST_EXCLUDE_URLS", "must_exclude_urls.txt")
URL_RE = r" URL: (\S+)$"
EXPECTED_DROPPED = 77_469


def gcs_listing():
    proc = subprocess.run(
        ["gcloud", "storage", "ls", "-l", GCS_GLOB],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gcs listing failed: {proc.stderr}")
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].startswith("gs://"):
            out[os.path.basename(parts[2])] = int(parts[0])
    return out


def resolve_scan_target(use_gcs: bool):
    if use_gcs:
        print(f"scanning GCS directly: {GCS_GLOB}")
        return GCS_GLOB
    remote = gcs_listing()
    local = {f: os.path.getsize(os.path.join(LOCAL_DIR, f))
             for f in os.listdir(LOCAL_DIR) if f.endswith(".parquet")}
    if local != remote:
        raise RuntimeError(
            "local slim mirror is stale vs GCS; re-sync it or rerun with --use-gcs")
    print(f"freshness check OK: {len(local)} files, "
          f"{sum(local.values()):,} bytes match GCS")
    return [os.path.join(LOCAL_DIR, f) for f in sorted(local)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gcs", action="store_true",
                        help="Scan the GCS glob instead of the local mirror")
    args = parser.parse_args()

    with open(URLS_FILE, encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    assert len(urls) == len(set(urls)), "duplicate urls in the must-exclude list"
    assert len(urls) == EXPECTED_DROPPED, (
        f"must-exclude list has {len(urls)} urls, expected {EXPECTED_DROPPED}")
    print(f"loaded {len(urls):,} must-exclude urls")

    target = resolve_scan_target(args.use_gcs)
    lf = pl.scan_parquet(target)

    # the exact expression from dynamic_data_loader.load_source_slice: tail-less
    # (noncjd) rows extract to null and MUST be kept via fill_null(True)
    url_tail = pl.col("metadata").str.extract(URL_RE, 1)
    keep_mask = (~url_tail.is_in(sorted(urls))).fill_null(True)

    counts = lf.select(
        pl.len().alias("rows_before"),
        keep_mask.sum().alias("rows_after"),
        url_tail.is_in(urls).sum().alias("matched_rows"),
        url_tail.is_null().sum().alias("tailless_rows"),
        url_tail.filter(url_tail.is_in(urls)).n_unique().alias("matched_unique_urls"),
    ).collect()
    row = counts.row(0, named=True)
    dropped = row["rows_before"] - row["rows_after"]

    print(f"rows before:          {row['rows_before']:,}")
    print(f"rows after keep-mask: {row['rows_after']:,}")
    print(f"rows dropped:         {dropped:,}")
    print(f"matched rows:         {row['matched_rows']:,}")
    print(f"matched unique urls:  {row['matched_unique_urls']:,}")
    print(f"tail-less rows kept:  {row['tailless_rows']:,}")

    failures = []
    if dropped != EXPECTED_DROPPED:
        failures.append(f"rows dropped {dropped:,} != {EXPECTED_DROPPED:,}")
    if row["matched_rows"] != EXPECTED_DROPPED:
        failures.append(f"matched rows {row['matched_rows']:,} != {EXPECTED_DROPPED:,}")
    if row["matched_unique_urls"] != EXPECTED_DROPPED:
        failures.append(
            f"matched unique urls {row['matched_unique_urls']:,} != {EXPECTED_DROPPED:,} "
            "(some must-exclude urls are absent from the slim data)")
    if dropped != row["matched_rows"]:
        failures.append(
            "keep-mask dropped a different row count than the direct is_in match; "
            "the fill_null(True) null handling is broken")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}")
        sys.exit(1)
    print("PASS: url exclusion drops exactly the must-exclude rows and nothing else")


if __name__ == "__main__":
    main()
