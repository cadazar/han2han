"""Exclusion registry: every v2 eval article vs today's slim training set.

One row per article in ANY v2 split: (master_url, split,
present_in_slim_training). Presence is url membership in the slim cjd url set;
the 33 slim rows with corrupted metadata (url tail unrecoverable) are resolved
exactly by text equality against the v2 article texts - any v2 article whose
text equals a corrupted slim row's original_text is marked present
(resolution='corrupted_text_match').

Also emits must_exclude_urls.txt (the present=true subset): tonight's runs
must not train on these. NOTE: the loader at tonight's pin commit has no
url-exclusion mechanism; enforcement is on the launch procedure.
"""

import argparse
import json
import os

os.environ.setdefault("POLARS_MAX_THREADS", str(os.cpu_count()))
import polars as pl

HOLDOUT_DIR = os.environ.get("HOLDOUT_DIR", ".")
SLIM_URLS = f"{HOLDOUT_DIR}/slim_urls.parquet"
SLIM_CORRUPTED = (f"{HOLDOUT_DIR}/"
                  "slim_urls_corrupted_metadata.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    slim = pl.read_parquet(SLIM_URLS)
    corrupted = pl.read_parquet(SLIM_CORRUPTED)
    print(f"slim urls: {slim.height:,}; corrupted slim rows: {corrupted.height}")

    parts = []
    total = 0
    for split in ("train", "validation", "test"):
        df = pl.read_parquet(
            os.path.join(args.v2_dir, f"korean_temporal_v2_{split}.parquet"))
        total += df.height
        parts.append(df.select(
            "master_url", "text", pl.lit(split).alias("split")))
    v2 = pl.concat(parts)
    if v2.height != total or v2["master_url"].n_unique() != v2.height:
        raise RuntimeError("v2 splits overlap or registry row count mismatch")

    v2 = v2.with_columns(
        pl.col("master_url").is_in(slim["url"]).alias("in_slim_by_url"),
        pl.col("text").is_in(corrupted["original_text"]).alias("in_slim_by_text"),
    ).with_columns(
        (pl.col("in_slim_by_url") | pl.col("in_slim_by_text"))
        .alias("present_in_slim_training"),
        pl.when(pl.col("in_slim_by_text") & ~pl.col("in_slim_by_url"))
        .then(pl.lit("corrupted_text_match"))
        .when(pl.col("in_slim_by_url")).then(pl.lit("url"))
        .otherwise(pl.lit("absent")).alias("resolution"),
    )

    registry = v2.select("master_url", "split", "present_in_slim_training",
                         "resolution")
    if registry.height != total:
        raise RuntimeError(f"registry rows {registry.height} != articles {total}")
    registry.write_parquet(os.path.join(args.outdir, "exclusion_registry.parquet"))
    registry.write_csv(os.path.join(args.outdir, "exclusion_registry.csv"))

    must = registry.filter(pl.col("present_in_slim_training"))
    with open(os.path.join(args.outdir, "must_exclude_urls.txt"), "w",
              encoding="utf-8") as f:
        for u in must["master_url"].to_list():
            f.write(u + "\n")

    unresolved_corrupted = corrupted.filter(
        ~pl.col("original_text").is_in(v2["text"])).height
    summary = {
        "registry_rows": registry.height,
        "present_in_slim": must.height,
        "absent_from_slim": registry.height - must.height,
        "resolved_by_corrupted_text_match":
            registry.filter(pl.col("resolution") == "corrupted_text_match").height,
        "corrupted_slim_rows_not_in_v2": unresolved_corrupted,
        "by_split": {k: int(v) for k, v in
                     registry.group_by("split").len().iter_rows()},
    }
    with open(os.path.join(args.outdir, "registry_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    print("NOTE: no url-exclusion mechanism exists in the loader at tonight's "
          "pin commit; must_exclude_urls.txt has to be enforced by the launch "
          "procedure")


if __name__ == "__main__":
    main()

