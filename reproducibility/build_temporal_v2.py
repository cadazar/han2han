"""Build the doubly-held-out korean_temporal_v2 splits.

Article pool: master articles matched by eval-side fragments (map mode) minus
articles flagged impure (purity mode). Text is the clean master title+"\n"+body
verbatim (no normalization, U+25A1 retained, zero <unk> by construction);
label is the 5-year bucket (year-1920)//5 over 1920-1999.

Content filters replicate the old builder's population definition where they
translate: must contain at least one Hanja char and pass the same
non-numeric-content ratio (>= 0.3). The old builder's 150-500 token filter
used the OLD tokenizer on old fragments and does not translate; character
length stats are reported instead (protocol delta, documented).

Per bucket the old realized counts are mirrored: 3000 train / 1000 validation
/ 1000 test. Buckets with fewer than 5000 available articles contribute
everything they have, split 60/20/20 by largest remainder; the shortfall is
reported per bucket. Sampling: numpy default_rng(42 + bucket) permutation
(deterministic; NEP 19 stable streams). Each url lands in exactly one split.
"""

import argparse
import json
import os
import re

os.environ.setdefault("POLARS_MAX_THREADS", str(os.cpu_count()))
import numpy as np
import polars as pl

MASTER = os.environ.get("CJD_MASTER", "cjd_master.parquet")
TARGETS = {"train": 3000, "validation": 1000, "test": 1000}
NON_TEXT_RE = re.compile(r"[\d\s\.\,\·\-\(\)\[\]\{\}\+\=\*\/]+")


def hanja_ratio(text):
    hanja = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return hanja / len(text) if text else 0.0


def sufficient_text(text, min_ratio=0.3):
    if not text:
        return False
    stripped = NON_TEXT_RE.sub("", text)
    return len(stripped) / len(text) >= min_ratio


def largest_remainder(n, fractions):
    raw = [n * f for f in fractions]
    base = [int(x) for x in raw]
    rest = n - sum(base)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:rest]:
        base[i] += 1
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", required=True)
    parser.add_argument("--impure", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    matched = pl.read_parquet(args.matches)
    impure = pl.read_parquet(args.impure)
    pool_urls = (
        matched.select(pl.col("matched_url").alias("url")).unique()
        .join(impure.select("url"), on="url", how="anti")
    )
    n_matched_urls = matched["matched_url"].n_unique()
    print(f"matched articles: {n_matched_urls:,}; impure removed: "
          f"{impure.height:,}; pool: {pool_urls.height:,}")

    master = (
        pl.scan_parquet(MASTER)
        .select(
            "url", "date",
            pl.col("date").dt.year().alias("year"),
            pl.when(pl.col("title").is_null()
                    | (pl.col("title").str.len_chars() == 0))
            .then(pl.col("body"))
            .otherwise(pl.col("title") + pl.lit("\n") + pl.col("body"))
            .alias("text"),
        )
        .join(pool_urls.lazy(), on="url", how="semi")
        .collect()
    )
    # master contains duplicate-url rows (reprint rows the slim build deduped);
    # keep the first in file order, mirroring build_cjd_master_shards.py
    n_with_dups = master.height
    master = master.unique(subset="url", keep="first", maintain_order=True)
    if n_with_dups != master.height:
        print(f"master duplicate-url rows among pool: "
              f"{n_with_dups - master.height:,} (deduped keep-first)")
    if master.height != pool_urls.height:
        raise RuntimeError(
            f"pool urls {pool_urls.height} vs master join {master.height}; "
            f"matched urls must all exist in master")

    counts = {"pool": master.height}
    in_range = master.filter((pl.col("year") >= 1920) & (pl.col("year") <= 1999))
    counts["dropped_year_out_of_range"] = master.height - in_range.height

    n_unk = in_range.filter(
        pl.col("text").str.contains("<unk>", literal=True)).height
    if n_unk:
        raise RuntimeError(f"{n_unk} master texts contain literal <unk>")

    texts = in_range["text"].to_list()
    keep_hanja = [any("\u4e00" <= c <= "\u9fff" for c in t) for t in texts]
    keep_content = [sufficient_text(t) for t in texts]
    keep = [h and c for h, c in zip(keep_hanja, keep_content)]
    counts["dropped_no_hanja"] = sum(1 for h in keep_hanja if not h)
    counts["dropped_low_text_content"] = sum(
        1 for h, c in zip(keep_hanja, keep_content) if h and not c)
    eligible = in_range.filter(pl.Series(keep)).with_columns(
        ((pl.col("year") - 1920) // 5).alias("label"))
    counts["eligible"] = eligible.height
    print(f"pool {counts['pool']:,} -> eligible {eligible.height:,} "
          f"(year drops {counts['dropped_year_out_of_range']}, "
          f"no-hanja {counts['dropped_no_hanja']}, "
          f"low-content {counts['dropped_low_text_content']})")

    split_parts = {s: [] for s in TARGETS}
    bucket_table = []
    for b in range(16):
        sub = eligible.filter(pl.col("label") == b).sort("url")
        avail = sub.height
        rng = np.random.default_rng(42 + b)
        perm = rng.permutation(avail)
        target_total = sum(TARGETS.values())
        if avail >= target_total:
            take = {s: TARGETS[s] for s in TARGETS}
        else:
            tr, va, te = largest_remainder(avail, [0.6, 0.2, 0.2])
            take = {"train": tr, "validation": va, "test": te}
        offset = 0
        for s in ("train", "validation", "test"):
            idx = perm[offset:offset + take[s]]
            offset += take[s]
            split_parts[s].append(sub[idx.tolist()].with_columns(
                pl.lit(s).alias("_split")))
        bucket_table.append({
            "bucket": b, "range": f"{1920 + 5 * b}-{1924 + 5 * b}",
            "available": avail,
            "train": take["train"], "validation": take["validation"],
            "test": take["test"],
            "shortfall": max(0, target_total - avail),
        })

    all_rows = []
    for s in TARGETS:
        df = pl.concat(split_parts[s])
        texts = df["text"].to_list()
        out = pl.DataFrame({
            "text": texts,
            "label": df["label"].cast(pl.Int64),
            "year": df["year"].cast(pl.Int64),
            "hanja_ratio": [hanja_ratio(t) for t in texts],
            "source": ["cjd_master"] * len(texts),
            "master_url": df["url"],
        })
        out.write_parquet(
            os.path.join(args.outdir, f"korean_temporal_v2_{s}.parquet"))
        all_rows.append(out.select("master_url").with_columns(
            pl.lit(s).alias("split")))
        print(f"{s}: {out.height:,} rows")

    registry_base = pl.concat(all_rows)
    if registry_base["master_url"].n_unique() != registry_base.height:
        raise RuntimeError("a master_url appears in more than one split")
    registry_base.write_parquet(os.path.join(args.outdir, "v2_url_splits.parquet"))

    bt = pl.DataFrame(bucket_table)
    bt.write_csv(os.path.join(args.outdir, "v2_bucket_table.csv"))
    print(bt)

    lens = eligible["text"].str.len_chars()
    counts["text_char_len"] = {
        "median": int(lens.median()), "p10": int(lens.quantile(0.1)),
        "p90": int(lens.quantile(0.9)), "max": int(lens.max()),
    }
    counts["total_selected"] = registry_base.height
    with open(os.path.join(args.outdir, "v2_build_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"counts": counts, "buckets": bucket_table,
                   "targets": TARGETS,
                   "versions": {"polars": pl.__version__,
                                "numpy": np.__version__}},
                  f, indent=2, ensure_ascii=False)
    print(f"selected {registry_base.height:,} articles across splits")


if __name__ == "__main__":
    main()

