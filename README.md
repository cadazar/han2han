# Han2Han

**Efficient language-specific character representation through script-aware pre-training for historical Korean text.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![EMNLP 2026 Findings](https://img.shields.io/badge/EMNLP%202026-Findings-b31b1b.svg)

Accepted to **Findings of the Association for Computational Linguistics: EMNLP 2026** ([citation](#citation)).

Han2Han is a 169M-parameter encoder-decoder model that learns **script-invariant**
Korean representations: it places a document written in Hanja (Sino-Korean
characters) and its Hangul transcription at the same point in embedding space.
It is trained with a language-aware recipe rather than scale -- jamo and
character-level embedding fusion, morpheme-aware denoising, and bidirectional
Hanja-Hangul transcription -- so that orthographic alignment is imposed as a
prior *before* distributional learning begins.

The goal is to make computational analysis of mixed-script Korean -- centuries
of intellectual production written in a blend of Hanja and Hangul -- accessible
to digital-humanities researchers without institutional ML infrastructure.

## Why this matters

Modern Korean text from roughly the 1900s-1980s mixes Hanja and Hangul freely,
and the two scripts can encode identical content (the Hanja `韓國語` and the
Hangul `한국어` are the *same word*). Off-the-shelf Korean models trained at much
larger scale segregate documents entirely by script: cross-script alignment does
not emerge from scale alone. Han2Han instead teaches the correspondence directly
through its pre-training objective, and the resulting etymological knowledge
transfers even when script cues are removed.

## Results

All numbers are from the paper. Han2Han is trained from scratch on **35B tokens**,
versus 131B-6.3T tokens for the KLUE baselines.

**KLUE benchmark** (YNAT macro-F1 %, KLUE dev; STS Spearman x100, KorSTS
dev/test under a matched fine-tuning loss; NLI accuracy %, KLUE dev):

| Model              | YNAT | STS (dev) | STS (test) | NLI  |
| ------------------ | ---- | --------- | ---------- | ---- |
| XLM-RoBERTa-base   | 80.6 | 81.7      | 79.0       | 75.2 |
| KLUE-BERT          | 84.9 | 87.7      | 82.5       | 80.3 |
| KLUE-RoBERTa-base  | 84.1 | 86.8      | 84.2       | 83.7 |
| KoELECTRA-base-v3  | 83.5 | 88.1      | 85.9       | 85.4 |
| SKT-KoBART-base-v1 | 86.0 | 84.9      | 81.4       | 77.6 |
| **Han2Han (169M)** | 84.5 | 84.4      | 75.1       | 74.7 |

Han2Han matches or approaches the baselines on YNAT despite the far smaller
training budget. On STS the dev score sits mid-range among the baselines, but on
the held-out test split the Korean-native encoders lead every Han2Han model, so
no parity claim is made on contemporary STS. NLI is the weakest dimension, consistent
with a pre-training mixture that deliberately prioritizes script-invariance and
morpheme structure over premise-hypothesis reasoning. Han2Han's claims live on
script-invariance and historical text, where the ordering reverses.

**Temporal classification** of mixed-script historical newspapers, evaluated on
*Hangul-only* inputs (the script cues are stripped, so the task tests transfer
across the script gap), on an **article-level held-out test split** whose
articles are excluded from pre-training by URL:

| Model              | Params | Test macro-F1 |
| ------------------ | ------ | ------------- |
| **Han2Han**        | 169M   | **76.0**      |
| T5Gemma 2          | 786M   | 54.3          |
| KLUE-BERT          | 110M   | 54.3          |
| SKT-KoBART-base-v1 | 124M   | 53.6          |
| KLUE-RoBERTa-base  | 110M   | 50.9          |
| KoELECTRA-base-v3  | 110M   | 49.4          |
| XLM-RoBERTa-base   | 270M   | 47.5          |

A **+21.7-point** margin over the strongest baselines. Each baseline received its
own optimizer and learning-rate sweep; every Han2Han model shares a single
fine-tuning recipe tuned once.

**Document-level script-invariance** on 1,124 hand-transcribed art-criticism
articles (CASASIA, 1920s-1940s); **k-NN** is bidirectional 1-NN cross-script
retrieval accuracy (higher = a document's nearest cross-script neighbor is
itself):

| Model                       | Params | Cos  | k-NN |
| --------------------------- | ------ | ---- | ---- |
| T5Gemma 2                   | 786M   | 0.84 | 0.03 |
| T5Gemma 2 + Han2Han recipe | 786M | 0.97 | 0.57 |
| **Han2Han (ours)**          | 169M   | 0.84 | **0.61** |

Han2Han 169M trained from scratch slightly exceeds the 4.7x-larger T5Gemma 2 +
recipe on the rank-based k-NN metric, on 1/57 the pre-training tokens. The recipe
is portable (it lifts T5Gemma 2 out of total script segregation), and the
architecture contributes alongside it, delivering the invariance at small scale. At the
lexical level, Han2Han reaches **0.87** centered top-1 Hanja->Hangul cognate
retrieval -- matched only by T5Gemma 2 + recipe (0.86), while 1.2-1.5B
decoder-only Korean LMs remain at or below 0.41.

## Architecture

Encoder-decoder, **640d / 2048 FFN, 18 + 18 layers** (169M parameters):

- **Subword feature fusion** (the core contribution): each token is decomposed
  into jamo and character 3-gram features. Hanja is transcribed to Hangul
  *before* decomposition, so cognates project to identical jamo n-gram features.
- **Attention**: grouped-query attention (4 query heads, 1 KV head, head_dim
  256; cross-attention 4 query / 2 KV heads), RoPE (theta 500k full / 10k
  sliding), QK-norm. Each 6-layer block is 5 sliding-window (window 128) + 1 full
  attention.
- **GeGLU** feed-forward, **SubLN** normalization, BF16 throughout.
- Weight tying: the encoder and decoder are tied completely, including their
  token embeddings, and the LM head is shared with those token embeddings. The
  jamo and character subtoken embeddings, all biases, and all scales remain
  untied; SubLN (above) then reintroduces variance between the two tied sides.

**Optimizer**: Muon with a Gram Newton-Schulz orthogonalization variant (AdamW
arm for 1D parameters), learning rate 2.66e-4, WSD schedule (10% warmup / 70%
constant / 20% sqrt cooldown), ~4.8M tokens per update, sequence length 2048.
All stochastic regularization (dropout, LayerDrop, label smoothing) is disabled:
the single-epoch / no-repetition regime makes the data and objective the source
of representational priors.

See `han2han-ul2-base-cooldown-v5e-eu.yaml` for the full configuration.

## The recipe

Han2Han is produced in three stages, each with a host-based TPU launch script.
The scripts assume `setup_tpu_host.sh` has been run on every pod-slice worker
(see *Installation*).

1. **Pre-training** -- UL2 R/X/S denoisers (2:2:1), morpheme-aware span masking,
   bidirectional Hanja<->Hangul transcription, and a temporal-continuation
   objective on year-stamped articles. ~35B tokens on a TPU v5e-64 slice
   (~35 wall-clock hours).

   ```bash
   CONFIG_FILE=han2han-ul2-base-cooldown-v5e-eu.yaml ./launch_han2han_ul2_host.sh
   ```

2. **Instruction tuning** -- chat-templated SFT with scheduled sampling and
   contrastive learning (no RLHF/DPO). TPU v6e-8 single host (~9 hours).

   ```bash
   CONFIG_FILE=configs/it-muon-stage_1.yaml ./launch_sft_host.sh
   ```

3. **Classifier fine-tuning** -- a classification head on the instruction-tuned
   backbone, fine-tuned per task. This is how all the downstream scores are
   produced (KLUE YNAT/STS/NLI and temporal classification), maintaining parity
   with the 2020s-era encoder-only baselines. Set `task:` in the config to
   `ynat`, `sts`, `nli`, or `temporal`; the shipped config runs temporal:

   ```bash
   CONFIG_FILE=configs/classifier_temporal.yaml ./launch_classifier_host.sh
   ```

   Generation-based classification is also possible directly through the
   instruction-tuned model (`finetune_sft.py`), but results trail the classifier
   head and would likely need substantially more pre-training tokens to close
   the gap.

Each config carries a `-local.yaml` sibling (`configs/it-muon-stage_1-local.yaml`,
`configs/classifier_temporal-local.yaml`) with a placeholder local checkpoint
path for single-host / CPU smoke runs.

## Checkpoints (PyTorch / Hugging Face)

The model has parallel Flax (training) and PyTorch (Hugging Face-compatible)
implementations. To convert a trained Flax checkpoint to an HF-loadable PyTorch
checkpoint:

```bash
python convert_flax_to_torch.py --ckpt_dir /path/to/flax/checkpoints \
    --output_dir converted/han2han-base
```

The converter runs an **unconditional numerical-equivalence harness** (encoder
and embedding parity, fp32 and BF16 forward passes under padded masks, and a
greedy rollout) and raises before writing the checkpoint if Flax and PyTorch
disagree. For a real-data sanity check, `compare_ppl_flax_torch.py` reports the
teacher-forced perplexity delta between the two frameworks on a held-out corpus:

```bash
python compare_ppl_flax_torch.py --flax_ckpt /path/to/flax/checkpoints \
    --pt_dir converted/han2han-base --data corpus.parquet --n 32 --max_length 256
```

Pre-trained weights will be released on the Hugging Face Hub.

## Demos

- `hanja_transcription_demo.ipynb` -- bidirectional Hanja<->Hangul transcription.
- `umap_comparisons.ipynb` -- the script-invariance UMAP visualizations from the
  paper (Han2Han vs. T5Gemma 2, before and after applying the recipe).

## Installation

Local use (checkpoint conversion, evaluation, notebooks):

```bash
pip install -r requirements.txt
pip install ./han2han_tools     # Rust extension for Hanja<->Hangul tooling; needs a Rust toolchain (https://rustup.rs)
```

`requirements.txt` is pinned to the reference environment. TPU pre-training uses
the JAX TPU wheels rather than the CPU/GPU `jax` pinned there; **`setup_tpu_host.sh`
is the single source of truth for the pod-slice environment** -- it provisions
Python, MeCab/KoNLPy, the Rust toolchain, JAX-on-TPU, and the data pipeline on
each worker. Run it on all workers before launching any stage:

```bash
gcloud alpha compute tpus tpu-vm ssh TPU_NAME --zone=ZONE \
    --worker=all --tunnel-through-iap --command='bash ~/setup_tpu_host.sh'
```

## Repository layout

| Path                                   | Purpose                                            |
| -------------------------------------- | -------------------------------------------------- |
| `modeling_han2han_flax.py`             | Flax model (training)                              |
| `modeling_han2han_pytorch.py`          | PyTorch / Hugging Face model                       |
| `han2han_config.py`                    | model configuration                                |
| `han2han_sampler.py`                   | generation / decoding                              |
| `convert_flax_to_torch.py`             | Flax -> PyTorch conversion + parity harness        |
| `compare_ppl_flax_torch.py`            | cross-framework perplexity check                   |
| `*_collator.py`, `dynamic_data_loader.py` | denoising, packing, and task-routing data pipeline |
| `subword_features.py`, `han2han_tokenizer.py` | jamo/character subword features + tokenizer  |
| `han2han_tools/`                       | Rust Hanja<->Hangul tooling                         |
| `*_callback.py`                        | evaluation callbacks (BLEU, ROUGE, temporal, etc.) |
| `configs/`                             | fine-tuning configs                                |
| `launch_*_host.sh`                     | host-based TPU launch scripts                      |
| `setup_tpu_host.sh`                    | TPU pod-slice environment provisioning             |

## Reproducibility: held-out evaluation

The temporal-classification results above use an article-level held-out split whose articles
are excluded from pre-training by URL at the data loader. The split builder, the exclusion
registry, the loader acceptance test, and a disclosure about the released checkpoint's residual
exposure are in [`reproducibility/`](reproducibility/).

## Citation

To appear in Findings of the Association for Computational Linguistics: EMNLP 2026.

```bibtex
@inproceedings{han2han2026,
  title     = {Han2Han: Efficient Language-Specific Character Representation
               through Script-Aware Pre-Training for Historical Text Analysis},
  author    = {Adams, Cellik and Jo, Eunkyoung and Kim, Ju-ae},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
