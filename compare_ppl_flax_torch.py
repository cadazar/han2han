"""Real-data equivalence + perplexity comparison: Flax vs converted PyTorch.

A complementary lens to the per-token greedy-rollout check in
`convert_flax_to_torch.py`. Where the conversion harness proves token-for-token
agreement on random inputs, this script measures agreement on the real
mixed-script historical-Korean manifold (the casasia eval articles), fed through
the model's actual training-time input structure (`BARTCollator`), and reports
teacher-forced perplexity for each framework plus their delta.

The PT-vs-Flax perplexity delta is the equivalence metric: the absolute PPL
reflects a pretraining denoiser reconstructing clean text and is not meaningful
on its own; the cross-framework delta is.

Usage:
    python compare_ppl_flax_torch.py \
        --flax_ckpt /path/to/han2han-base/checkpoints \
        --pt_dir converted_checkpoints/han2han-base \
        --data casasia_articles.parquet --n 32 --max_length 256
"""

import argparse
import os

import numpy as np
import pandas as pd


def _ce_ppl(logits, labels, pad_token_id):
    """Per-token cross-entropy and perplexity over non-ignored label positions.

    logits: [B, T, V] float32 numpy. labels: [B, T] int numpy with -100 or
    pad_token_id marking ignored positions.
    """
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(logits).sum(axis=-1))
    gathered = np.take_along_axis(
        logits, np.clip(labels, 0, logits.shape[-1] - 1)[..., None], axis=-1
    )[..., 0]
    token_logprob = gathered - logsumexp
    mask = (labels != -100) & (labels != pad_token_id)
    nll = -(token_logprob * mask).sum() / max(mask.sum(), 1)
    return float(nll), float(np.exp(nll)), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flax_ckpt", required=True)
    ap.add_argument("--pt_dir", required=True)
    ap.add_argument("--data", default="casasia_articles.parquet")
    ap.add_argument("--tokenizer_path", default="han2han_v2_tokenizer")
    ap.add_argument("--text_col", default="original_text")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=734)
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import torch
    import jax.numpy as jnp
    import flax.nnx as nnx
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import modeling_han2han_pytorch  # registers Han2Han AutoClasses
    import register_han2han  # noqa: F401  (registers Han2HanTokenizer)
    from h2hcollator import BARTCollator
    from convert_flax_to_torch import load_orbax_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"tokenizer {args.tokenizer_path}: vocab_size={tokenizer.vocab_size}")
    pad_id = tokenizer.pad_token_id

    # canonical Original->Original collation: no corruption, deterministic. feeds
    # the denoiser its training-time structure (script sentinels, sentence
    # boundaries) so the comparison is on-distribution.
    collator = BARTCollator(
        tokenizer=tokenizer, rng=np.random.default_rng(args.seed),
        model_max_length=args.max_length, max_length=args.max_length,
        hangul_decoder=False, infilling_ratio=0.0, sentence_permutation=False,
        hangul_only=False, use_morpheme_masking=0.0,
    )

    df = pd.read_parquet(args.data)
    texts = [str(t) for t in df[args.text_col].tolist()[: args.n]]
    examples = [{"original_text": t, "metadata": ""} for t in texts]
    inputs = collator(examples, cooldown_phase=True)

    input_ids = np.asarray(inputs["input_ids"])
    attention_mask = np.asarray(inputs["attention_mask"])
    decoder_input_ids = np.asarray(inputs["decoder_input_ids"])
    decoder_attention_mask = np.asarray(inputs["decoder_attention_mask"])
    labels = np.asarray(inputs["labels"])
    print(f"collated {input_ids.shape[0]} examples, enc_len={input_ids.shape[1]} dec_len={labels.shape[1]}")

    # Flax (fp32) for a clean equivalence delta (bf16 would add its own noise).
    _, _, flax_model = load_orbax_checkpoint(os.path.abspath(args.flax_ckpt),
                                             sorted(int(c) for c in os.listdir(args.flax_ckpt))[-1])
    flax_model.eval()
    flax_logits = np.asarray(flax_model(
        input_ids=jnp.array(input_ids), attention_mask=jnp.array(attention_mask),
        decoder_input_ids=jnp.array(decoder_input_ids),
        decoder_attention_mask=jnp.array(decoder_attention_mask),
        return_dict=True, rngs=nnx.Rngs(0),
    ).logits).astype(np.float32)

    pt = AutoModelForSeq2SeqLM.from_pretrained(args.pt_dir, dtype=torch.float32)
    pt.eval()
    with torch.no_grad():
        pt_logits = pt(
            input_ids=torch.from_numpy(input_ids).long(),
            attention_mask=torch.from_numpy(attention_mask).long(),
            decoder_input_ids=torch.from_numpy(decoder_input_ids).long(),
            decoder_attention_mask=torch.from_numpy(decoder_attention_mask).long(),
            return_dict=True,
        ).logits.float().cpu().numpy()

    nll_f, ppl_f, mask = _ce_ppl(flax_logits, labels, pad_id)
    nll_p, ppl_p, _ = _ce_ppl(pt_logits, labels, pad_id)

    am_f = flax_logits.argmax(-1)
    am_p = pt_logits.argmax(-1)
    argmax_agree = float((am_f[mask] == am_p[mask]).mean())
    drift = np.abs(flax_logits - pt_logits)

    print("\n=== Flax vs PyTorch real-data equivalence (casasia, fp32, teacher-forced) ===")
    print(f"  scored label tokens: {int(mask.sum())}")
    print(f"  Flax  NLL={nll_f:.5f}  PPL={ppl_f:.4f}")
    print(f"  PT    NLL={nll_p:.5f}  PPL={ppl_p:.4f}")
    print(f"  |dNLL|={abs(nll_f-nll_p):.3e}  |dPPL|={abs(ppl_f-ppl_p):.3e}  rel_dPPL={abs(ppl_f-ppl_p)/max(ppl_f,1e-9):.3e}")
    print(f"  per-token argmax agreement (scored positions): {100*argmax_agree:.2f}%")
    print(f"  logit |diff|: median={np.median(drift):.3e} p99={np.percentile(drift,99):.3e} max={drift.max():.3e}")


if __name__ == "__main__":
    main()
