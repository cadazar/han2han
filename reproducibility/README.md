# Reproducibility: held-out temporal split and pre-training exclusion

These scripts document the article-level held-out protocol reported in Section 5.3 of the
paper, and the URL exclusion mechanism that keeps evaluation articles out of pre-training.

## The protocol

`korean_temporal_v2` is built from 77,920 clean Chojoongdong articles, partitioned
46,752 / 15,584 / 15,584 into train / validation / test across sixteen five-year buckets
covering 1920-1999. Every article in the split -- all three partitions -- is registered in a
URL exclusion list. The pre-training data loader drops those URLs before any training batch
is drawn, so no pre-training run can reach evaluation text.

## Scripts

| script | what it does |
| ------ | ------------ |
| `build_temporal_v2.py` | Builds the doubly-held-out `korean_temporal_v2` splits from the master article table. Text is the clean master `title + "\n" + body` verbatim; the label is the 5-year bucket. |
| `make_exclusion_registry.py` | Emits one row per evaluation article recording whether its URL is present in the training set, and writes `must_exclude_urls.txt` -- the subset the loader must drop. Resolves rows with corrupted URL metadata by exact text equality. |
| `verify_url_exclusion_count.py` | Acceptance test. Applies the loader's exact keep-mask expression to the full training bundle and asserts that every must-exclude URL is dropped, that none is missing from the data, and that no non-newspaper row is affected. |

## Paths

Corpus paths are read from the environment rather than hardcoded, because the underlying
newspaper corpus cannot be redistributed (see the paper's Ethics Statement and Appendix A.3):

```
CJD_MASTER          master article parquet
HOLDOUT_DIR         directory holding slim_urls.parquet and must_exclude_urls.txt
SLIM_LOCAL_DIR      local mirror of the training bundle
GCS_PREFIX          remote prefix for the training bundle
SLIM_GLOB           glob for the training shards
MUST_EXCLUDE_URLS   path to must_exclude_urls.txt
```

## Scope and a disclosure

The exclusion mechanism post-dates the released V1 checkpoint. As stated in Section 5.3 of
the paper, V1's hold-out was re-drawn when pre-training entered its final cooldown phase, so
some evaluation articles may have received gradients as unlabelled denoising text during the
last 20% of that run, and which ones cannot be recovered. Every ablation arm reported in
Appendix D excludes the evaluation articles by construction and reproduces the result at full
strength. The registry is now wired into the loader, so no future run can reach these articles.
