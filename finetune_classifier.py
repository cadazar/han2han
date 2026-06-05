#!/usr/bin/env python3
# coding: utf-8
"""
Classification fine-tuning for Han2Han using Flax-native classifier head.

Single-task script: loads raw parquet data via get_local_sft_datasets(),
tokenizes with prepare_for_encoder/decoder, and trains a linear classifier
on top of mean-pooled decoder hidden states.

Reuses optimizer, LR schedule, mesh, and checkpoint infrastructure from the
pretraining/SFT scripts.
"""

import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout,
    force=True,
)

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.WARNING)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
if os.getenv('ENABLE_JAX_DISTRIBUTED'):
    try:
        import jax
        jax.distributed.initialize()
        print("JAX distributed training initialized")
    except RuntimeError as e:
        import jax
        if "already been initialized" not in str(e):
            print(f"JAX distributed initialization failed: {e}")
else:
    import jax

import jax.numpy as jnp
from flax import nnx
import numpy as np
import argparse
import time
import yaml
import optax
from typing import Dict, List, Optional
from scipy import stats as scipy_stats
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer

from han2han_config import Han2HanConfig
from modeling_han2han_flax import FlaxHan2HanForSequenceClassification
from checkpoint_utils import restore_checkpoint, load_config_from_checkpoint
from sharding_utils import setup_mesh_and_sharding, get_data_layout, shard_batch_to_devices
from dynamic_data_loader import get_local_sft_datasets
from logging_utils import log_from_main_process
from task_prompts import sample_task_prompt
from optimizer import (
    create_optimizer,
    create_learning_rate_schedule,
)
from train_han2han import get_config as _get_pretraining_config

logger = logging.getLogger(__name__)

TASK_CONFIGS = {
    'sts': {
        'sft_filter': 'sts',
        'dataset_name': 'kornlu_sts',
        'num_labels': 1,
        'task_type': 'regression',
        'primary_metric': 'spearman',
        'prompt_key': 'sts',
    },
    'nli': {
        'sft_filter': 'nli',
        'dataset_name': 'klue_nli',
        'num_labels': 3,
        'task_type': 'classification',
        'primary_metric': 'accuracy',
        'prompt_key': 'nli',
    },
    'ynat': {
        'sft_filter': 'topic_classification',
        'dataset_name': 'klue_ynat',
        'num_labels': 7,
        'task_type': 'classification',
        'primary_metric': 'macro_f1',
        'prompt_key': 'classification',
    },
    'temporal': {
        'sft_filter': 'temporal_classification',
        'dataset_name': 'korean_temporal',
        'num_labels': 16,
        'task_type': 'classification',
        'primary_metric': 'macro_f1',
        'prompt_key': 'temporal_classification',
    },
}


def build_label_map(dataset, label_field: str) -> Dict[str, int]:
    """Build a string-to-int label mapping from unique values in the dataset."""
    unique_labels = sorted(set(dataset[label_field]))
    return {label: idx for idx, label in enumerate(unique_labels)}


def _resolve_chat_token_ids(tokenizer) -> Dict[str, int]:
    """Resolve the 5 chat-template token ids from the tokenizer.

    Mirrors ChatSFTCollator._init_chat_token_ids: prefer the dedicated
    convenience attrs set by map_chat_tokens(), fall back to convert_tokens_to_ids,
    and fail loudly if any chat token is missing.
    """
    def _resolve(attr_name, token_str):
        if hasattr(tokenizer, attr_name):
            tid = getattr(tokenizer, attr_name)
            if tid is not None:
                return tid
        tid = (
            tokenizer.convert_tokens_to_ids(token_str)
            if hasattr(tokenizer, 'convert_tokens_to_ids') else None
        )
        unk = getattr(tokenizer, 'unk_token_id', None)
        if tid is None or tid == unk:
            raise ValueError(
                f"chat tokenization requires {token_str!r} in the tokenizer "
                f"vocabulary (got id={tid}, unk_id={unk}). For Han2Han, "
                f"regenerate with prepare_multilingual_tokenizer.py. For the "
                f"retired Han2HanTokenizer, call map_chat_tokens() after "
                f"loading to alias chat tokens onto sentinel slots."
            )
        return tid

    return {
        'system': _resolve('system_token_id', '<|system|>'),
        'user': _resolve('user_token_id', '<|user|>'),
        'assistant': _resolve('assistant_token_id', '<|assistant|>'),
        'end_of_turn': _resolve('end_of_turn_token_id', '<|end_of_turn|>'),
        'think': _resolve('think_token_id', '<|think|>'),
    }


def tokenize_for_classifier(
    examples: list,
    tokenizer,
    task: str,
    label_map: Optional[Dict[str, int]],
    max_encoder_length: int,
    max_decoder_length: int,
    prompt_key: Optional[str] = None,
    chat_token_ids: Optional[Dict[str, int]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Convert raw examples to tokenized classifier inputs.

    Returns list of dicts with input_ids, attention_mask, decoder_input_ids,
    decoder_attention_mask, and labels (numeric).

    When ``chat_token_ids`` is provided, the encoder is built as
    ``<|system|>{prompt}<|user|>{text}<|end_of_turn|>`` and the decoder as
    ``<|assistant|>{decoder_text}<|end_of_turn|>``, matching the format the
    model saw under finetune_sft.py with chat_sft_collator.py. Otherwise the
    pretraining-era prepare_for_encoder/decoder format is used.
    """
    use_chat = chat_token_ids is not None
    # pad id for the chat branch only; unified branch keeps its original 0-fill
    # so PT-vs-IT comparison on that path stays bit-identical to prior runs
    chat_pad_id = tokenizer.pad_token_id if use_chat else 0

    def pad_to(arr, length, pad_val=0):
        arr = list(arr)
        if len(arr) >= length:
            return arr[:length]
        return arr + [pad_val] * (length - len(arr))

    processed = []
    for ex in examples:
        if task == 'sts':
            encoder_text = ex['sentence1']
            decoder_text = ex['sentence2']
            label = float(ex['rounded_score'])
        elif task == 'temporal':
            encoder_text = ex['text']
            decoder_text = None
            label = int(ex['label'])
        else:
            encoder_text = ex['input_text']
            decoder_text = None
            label_str = ex['target_text']
            if label_str not in label_map:
                log_from_main_process(logger, 'warning',f"Unknown label '{label_str}', skipping")
                continue
            label = label_map[label_str]

        prompt_text = None
        if prompt_key:
            prompt_text, _ = sample_task_prompt(prompt_key)

        if use_chat:
            system_id = chat_token_ids['system']
            user_id = chat_token_ids['user']
            assistant_id = chat_token_ids['assistant']
            end_of_turn_id = chat_token_ids['end_of_turn']

            # raw tokenization without script special tokens; chat tokens
            # are inserted manually below
            usr_ids = tokenizer(encoder_text, add_special_tokens=False).input_ids
            sys_ids = (
                tokenizer(prompt_text, add_special_tokens=False).input_ids
                if prompt_text else []
            )
            dec_content = decoder_text if decoder_text is not None else encoder_text
            rsp_ids = tokenizer(dec_content, add_special_tokens=False).input_ids

            enc_ids = []
            if sys_ids:
                enc_ids.append(system_id)
                enc_ids.extend(sys_ids)
            enc_ids.append(user_id)
            enc_ids.extend(usr_ids)
            enc_ids.append(end_of_turn_id)
            if len(enc_ids) > max_encoder_length:
                enc_ids = enc_ids[: max_encoder_length - 1] + [end_of_turn_id]
            enc_mask = [1] * len(enc_ids)

            dec_ids = [assistant_id] + rsp_ids + [end_of_turn_id]
            if len(dec_ids) > max_decoder_length:
                dec_ids = dec_ids[: max_decoder_length - 1] + [end_of_turn_id]
            dec_mask = [1] * len(dec_ids)
        else:
            if prompt_text:
                encoder_text = f"{prompt_text} {encoder_text}"

            enc = tokenizer.prepare_for_encoder(
                encoder_text,
                padding=False,
                truncation=True,
                max_length=max_encoder_length,
                return_tensors=None,
            )
            enc_ids = enc['input_ids']
            enc_mask = enc['attention_mask']

            dec = tokenizer.prepare_for_decoder(
                decoder_text if decoder_text is not None else encoder_text,
                padding=False,
                truncation=True,
                max_length=max_decoder_length,
                return_tensors=None,
            )
            dec_ids = dec['input_ids']
            dec_mask = dec['attention_mask']

        processed.append({
            'input_ids': np.array(pad_to(enc_ids, max_encoder_length, chat_pad_id), dtype=np.int32),
            'attention_mask': np.array(pad_to(enc_mask, max_encoder_length), dtype=np.int32),
            'decoder_input_ids': np.array(pad_to(dec_ids, max_decoder_length, chat_pad_id), dtype=np.int32),
            'decoder_attention_mask': np.array(pad_to(dec_mask, max_decoder_length), dtype=np.int32),
            'labels': np.float32(label) if task == 'sts' else np.int32(label),
        })

    return processed


def create_batches_from_examples(examples: List[Dict], batch_size: int):
    """Stack individual examples into batched numpy arrays."""
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        if len(chunk) < batch_size:
            continue
        batch = {
            key: np.stack([ex[key] for ex in chunk], axis=0)
            for key in chunk[0].keys()
        }
        yield batch


@nnx.jit(static_argnums=3, donate_argnums=2)
def classification_train_step(
    model: FlaxHan2HanForSequenceClassification,
    optimizer: nnx.Optimizer,
    batch: Dict[str, jnp.ndarray],
    task_type: str,
    dropout_rngs: nnx.Rngs,
    progress: jnp.ndarray,
) -> jnp.ndarray:
    """Single classification training step. ``progress`` is the normalized
    training progress in [0.0, 1.0] used to drive the progress-based LR /
    WD schedules baked into the optimizer chain."""

    def loss_fn(model_local, rngs_local):
        output = model_local(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=batch['decoder_input_ids'],
            decoder_attention_mask=batch['decoder_attention_mask'],
            deterministic=False,
            rngs=rngs_local,
        )
        logits = output.logits
        if task_type == 'regression':
            loss = jnp.mean((logits.squeeze(-1) - batch['labels']) ** 2)
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch['labels']
            ).mean()
        return loss

    grad_fn = nnx.value_and_grad(
        loss_fn, argnums=nnx.DiffState(0, optimizer.wrt)
    )
    loss, grad = grad_fn(model, dropout_rngs)
    optimizer.update(model, grad, progress=progress)
    return loss


@nnx.jit
def classification_eval_step(
    model: FlaxHan2HanForSequenceClassification,
    batch: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Single classification eval forward pass."""
    output = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        decoder_input_ids=batch['decoder_input_ids'],
        decoder_attention_mask=batch['decoder_attention_mask'],
        deterministic=True,
    )
    return output.logits


def evaluate(
    model: FlaxHan2HanForSequenceClassification,
    eval_examples: List[Dict],
    batch_size: int,
    task_type: str,
    mesh,
) -> Dict[str, float]:
    """Run evaluation and compute task-specific metrics."""
    all_preds = []
    all_labels = []

    for batch in create_batches_from_examples(eval_examples, batch_size):
        labels_np = batch.pop('labels')

        with mesh:
            batch_sharded = shard_batch_to_devices(batch, mesh)
            logits = classification_eval_step(model, batch_sharded)

        logits = np.concatenate(
            [s.data for s in logits.addressable_shards], axis=0
        )

        if task_type == 'regression':
            preds = logits.squeeze(-1)
        else:
            preds = np.argmax(logits, axis=-1)

        all_preds.append(preds)
        all_labels.append(labels_np)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {}
    if task_type == 'regression':
        if np.std(all_preds) < 1e-8:
            log_from_main_process(logger, 'warning',"Predictions have near-zero variance -- likely collapsed")
            metrics['pearson'] = 0.0
            metrics['spearman'] = 0.0
        else:
            metrics['pearson'] = float(scipy_stats.pearsonr(all_preds, all_labels).statistic)
            metrics['spearman'] = float(scipy_stats.spearmanr(all_preds, all_labels).statistic)
    else:
        metrics['accuracy'] = float(accuracy_score(all_labels, all_preds))
        metrics['macro_f1'] = float(f1_score(all_labels, all_preds, average='macro'))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Classification fine-tuning for Han2Han (Flax)")

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="han2han_v2_tokenizer")
    parser.add_argument("--output_dir", type=str, default="classifier_output")
    parser.add_argument("--task", type=str, default=None, choices=list(TASK_CONFIGS.keys()))

    parser.add_argument("--data_dir", type=str, default="task_data")
    parser.add_argument("--data_bucket", type=str, default=None)
    parser.add_argument("--max_encoder_length", type=int, default=256)
    parser.add_argument("--max_decoder_length", type=int, default=64)

    parser.add_argument("--use_task_prompts", action="store_true", default=False)
    parser.add_argument("--collator", type=str, default="unified",
                        choices=["unified", "chat"],
                        help="Which tokenization scheme to use. 'unified' (default): "
                             "pretraining-era prepare_for_encoder/decoder with "
                             "script-tag boundaries (use for PT checkpoints). "
                             "'chat': ChatML-style boundaries (<|system|>, "
                             "<|user|>, <|assistant|>, <|end_of_turn|>), matching "
                             "finetune_sft.py + chat_sft_collator.py. Use for "
                             "IT checkpoints so the model sees the format it was "
                             "fine-tuned on. For the retired Han2Han tokenizer, "
                             "chat tokens are auto-aliased onto sentinel slots via "
                             "map_chat_tokens().")

    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "constant", "linear", "rsqrt"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.15)
    parser.add_argument("--constant_ratio", type=float, default=0.0)
    parser.add_argument("--lr_cooldown_ratio", type=float, default=0.0)
    parser.add_argument("--lr_cooldown_type", type=str, default="linear",
                        choices=["linear", "sqrt"],
                        help="Cooldown shape: linear or sqrt (1-sqrt(x), fast initial decay)")
    parser.add_argument("--clipnorm", type=float, default=0.0,
                        help="Global gradient clip norm (0 = disabled)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--optimizer", type=str, default="muon",
                        choices=["muon"])
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer_state_dtype", type=str, default=None,
                        choices=["bfloat16", "float32", "float16"],
                        help="Storage dtype for optimizer moment accumulators "
                             "(mu_dtype / accumulator_dtype). Defaults to --model_dtype.")
    parser.add_argument("--adafactor_beta2_cap", type=float, default=0.999)
    parser.add_argument("--adafactor_decay_rate", type=float, default=0.8,
                        help="Exponent for Adafactor factored 2nd-moment decay schedule "
                             "1 - (t+1)^(-decay_rate) (default: 0.8, paper value)")
    parser.add_argument("--adafactor_constant_beta2", action="store_true")
    parser.add_argument("--adafactor_momentum", type=float, default=0.0)
    parser.add_argument("--adafactor_burnin_steps", type=int, default=0)
    parser.add_argument("--use_param_block_rms", action="store_true", default=True)
    parser.add_argument("--no_param_block_rms", dest="use_param_block_rms", action="store_false")

    # adamw / lion / muon adam-arm betas
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="First-moment decay. AdamW: b1. Lion: sign-update momentum. "
                             "Muon Adam-arm: b1. Adafactor variants ignore this.")
    parser.add_argument("--beta2", type=float, default=0.99,
                        help="Second-moment decay. AdamW: b2. Lion: tracked momentum EMA. "
                             "Muon Adam-arm: b2. Adafactor variants ignore this.")

    # muon specific
    parser.add_argument("--muon_beta", type=float, default=0.95,
                        help="Momentum decay for the Muon arm (orthogonalized momentum).")
    parser.add_argument("--muon_gamma", type=float, default=0.18,
                        help="DeepSeek V4 / Moonlight update rescaling factor (muon only).")
    parser.add_argument("--muon_hybrid_ns", type=lambda x: x.lower() == 'true', default=False,
                        help="Use DeepSeek V4's 10-step hybrid Newton-Schulz schedule.")
    parser.add_argument("--muon_adam_wd_ratio", type=float, default=0.0,
                        help="Multiplier on weight_decay for the Muon AdamW arm.")
    parser.add_argument("--muon_adam_wd_skip_1d", type=lambda x: x.lower() == 'true', default=True,
                        help="If True, Muon Adam-arm WD skips 1D params (norm scales, biases).")
    parser.add_argument("--ns_variant", type=str, default="standard",
                        choices=["standard", "gram"],
                        help="Newton-Schulz inner algorithm: standard (Keller-Jordan quintic) "
                             "or gram (Dao 2026, iterate on n x n Gram matrix).")
    parser.add_argument("--gram_ns_reset_iters", type=str, default="2",
                        help="Comma-separated iteration indices BEFORE which to restart Gram NS.")
    parser.add_argument("--gram_ns_dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Working dtype for Gram NS inner R, Q, Z matrices.")

    # checkpoint averaging (applied before training begins)
    parser.add_argument("--average_last_n", type=int, default=0,
                        help="Average the last N checkpoints before fine-tuning (0 = disabled)")
    parser.add_argument("--average_steps", type=int, nargs='+', default=None,
                        help="Average specific checkpoint steps before fine-tuning")
    parser.add_argument("--average_weights", type=float, nargs='+', default=None,
                        help="Per-checkpoint weights for averaging (normalized to sum=1)")

    parser.add_argument("--classf_pdrop", type=float, default=0.1)
    parser.add_argument("--classifier_head_type", type=str, default="linear",
                        choices=["linear", "mlp"])
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")

    parser.add_argument("--attndrop", type=float, default=0.0)
    parser.add_argument("--resdrop", type=float, default=0.0)
    parser.add_argument("--embddrop", type=float, default=0.0)
    parser.add_argument("--embedding_dropout_rate", type=float, default=0.0)
    parser.add_argument("--layerdrop", type=float, default=0.0)
    parser.add_argument("--cross_attn_pdrop", type=float, default=0.0)

    parser.add_argument("--model_dtype", "--dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16"])

    # tiered weight-decay / adaptive WD / trust-ratio / fromage knobs forwarded to
    # create_optimizer. Pretraining-only flags (max_tokens, phase2_*, etc.) are
    # backfilled from train_han2han's parser below, so they don't need
    # to be redeclared here -- declare what you actually want overridable from YAML.
    parser.add_argument("--mlp_weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_weight_decay", type=float, default=0.0)
    parser.add_argument("--lm_head_weight_decay", type=float, default=0.0)
    parser.add_argument("--norm_weight_decay", type=float, default=0.0)
    parser.add_argument("--bias_weight_decay", type=float, default=0.0)
    parser.add_argument("--proportional_weight_decay", action="store_true", default=False)
    parser.add_argument("--adaptive_wd", action="store_true", default=False)
    parser.add_argument("--wd_base", type=float, default=0.1)
    parser.add_argument("--wd_min_value", type=float, default=1e-6)
    parser.add_argument("--wd_max_value", type=float, default=1.0)
    parser.add_argument("--wd_target_rms", type=float, default=1.0)
    parser.add_argument("--wd_scale_metric", type=str, default="rms")
    parser.add_argument("--wd_scale_mult", type=float, default=0.01)
    parser.add_argument("--wd_bias_mult", type=float, default=0.001)
    parser.add_argument("--wd_warmup_scales", action="store_true", default=False)
    parser.add_argument("--wd_warmup_shape", type=str, default="sigmoid")
    parser.add_argument("--use_trust_ratio", type=str, default="false")
    parser.add_argument("--trust_ratio_min_norm", type=float, default=1e-6)
    parser.add_argument("--use_fromage_style", type=str, default="false")
    parser.add_argument("--tie_input_output_embeddings", type=bool, default=False)

    args = parser.parse_args()

    # backfill pretraining-only args (max_tokens, batch_warmup_tokens,
    # phase2_*, etc.) with their pretraining-script defaults BEFORE the YAML merge,
    # so they (a) exist as real attributes on args (create_optimizer / lr schedule
    # won't AttributeError) and (b) are still overridable from this script's YAML
    # config like any classifier-native arg.
    _pretrain_parser = _get_pretraining_config(return_parser_only=True)
    _backfilled = []
    for _action in _pretrain_parser._actions:
        if _action.dest == 'help' or hasattr(args, _action.dest):
            continue
        setattr(args, _action.dest, _action.default)
        _backfilled.append(_action.dest)
    if _backfilled:
        log_from_main_process(logger, 'info',
            f"Backfilled {len(_backfilled)} pretraining-only args with their defaults; "
            f"these are now overridable via YAML/CLI.")

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # type coercion driven by parser action types across both parsers (covers
        # classifier-native and pretraining-backfilled args).
        _all_actions = {
            a.dest: a
            for a in list(parser._actions) + list(_pretrain_parser._actions)
            if a.dest != 'help'
        }
        for key, val in config.items():
            if hasattr(args, key):
                action = _all_actions.get(key)
                if action is not None and action.type is not None and isinstance(val, str):
                    val = action.type(val)
                setattr(args, key, val)

    if args.model_path is None:
        parser.error("--model_path is required (via CLI or config file)")
    if args.task is None:
        parser.error("--task is required (via CLI or config file)")

    task_cfg = TASK_CONFIGS[args.task]
    log_from_main_process(logger, 'info',f"Task: {args.task} ({task_cfg['task_type']}, {task_cfg['num_labels']} labels)")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # opt-in chat-template tokenization (matches finetune_sft.py + chat_sft_collator.py).
    # han2han_tokenizer (retired pretraining tokenizer) doesn't have chat tokens
    # baked into its SPM model; alias them onto the first 5 sentinel slots so
    # existing IT checkpoints can be evaluated with the format they were trained on.
    chat_token_ids = None
    if args.collator == "chat":
        if hasattr(tokenizer, "map_chat_tokens"):
            tokenizer.map_chat_tokens()
        chat_token_ids = _resolve_chat_token_ids(tokenizer)
        log_from_main_process(logger, 'info',
            f"Chat-template tokenization enabled; chat token ids: {chat_token_ids}")

    # load model config and override for classification
    model_config_dict = load_config_from_checkpoint(args.model_path)
    model_config = Han2HanConfig.from_dict(model_config_dict)
    model_config.num_labels = task_cfg['num_labels']
    model_config.attndrop = args.attndrop
    model_config.resdrop = args.resdrop
    model_config.embd_pdrop = args.embddrop
    model_config.embedding_dropout_rate = args.embedding_dropout_rate
    model_config.layerdrop = args.layerdrop
    model_config.cross_attn_pdrop = args.cross_attn_pdrop
    model_config.classf_pdrop = args.classf_pdrop
    model_config.classifier_head_type = args.classifier_head_type

    # --model_dtype is activation/compute precision; --param_dtype (backfilled
    # from train_han2han's parser) is latent-weight storage. If
    # omitted, storage matches compute. Threading param_dtype through to the
    # classifier head matters for full fine-tuning + Muon: a mixed-dtype param
    # tree breaks optax.partition/MultiSteps' lax.cond.
    dtype = jnp.bfloat16 if args.model_dtype == "bfloat16" else jnp.float32
    param_dtype = getattr(jnp, args.param_dtype) if args.param_dtype else dtype
    opt_state_dtype_name = args.optimizer_state_dtype or args.model_dtype
    log_from_main_process(logger, 'info',
        f"Precision: activation={dtype.__name__}, param={param_dtype.__name__}, "
        f"optimizer_state={opt_state_dtype_name}")

    # mesh setup
    args.parallelism_strategy = "data_parallel"
    mesh, _, parallelism_config = setup_mesh_and_sharding(args)
    n_devices = jax.device_count()
    devices_per_host = n_devices // jax.process_count()
    data_layout = get_data_layout(args.batch_size, devices_per_host)
    global_batch_size = data_layout['global_batch_size']
    grad_accum = max(int(getattr(args, 'gradient_accumulation_steps', 1)), 1)
    effective_global_batch_size = global_batch_size * grad_accum
    log_from_main_process(logger, 'info',
        f"Devices: {n_devices}, micro batch size: {global_batch_size}, "
        f"grad_accum: {grad_accum}, effective batch size: {effective_global_batch_size}")

    # load raw data
    train_datasets, _, _ = get_local_sft_datasets(
        base_dir=args.data_dir,
        sft_tasks=task_cfg['sft_filter'],
        split="train",
        data_bucket=args.data_bucket,
        host_idx=jax.process_index(),
        num_hosts=jax.process_count(),
    )
    val_datasets, _, _ = get_local_sft_datasets(
        base_dir=args.data_dir,
        sft_tasks=task_cfg['sft_filter'],
        split="validation",
        data_bucket=args.data_bucket,
    )

    ds_name = task_cfg['dataset_name']
    if ds_name not in train_datasets:
        raise ValueError(
            f"Dataset '{ds_name}' not found. Available: {list(train_datasets.keys())}"
        )

    train_ds = train_datasets[ds_name]
    val_ds = val_datasets.get(ds_name)
    log_from_main_process(logger, 'info',f"Train: {len(train_ds)} examples" + (f", Val: {len(val_ds)}" if val_ds else ""))

    # build label map for classification tasks (temporal uses int labels directly)
    label_map = None
    if task_cfg['task_type'] == 'classification' and args.task != 'temporal':
        label_map = build_label_map(train_ds, 'target_text')
        log_from_main_process(logger, 'info',f"Label map: {label_map}")
        if len(label_map) != task_cfg['num_labels']:
            log_from_main_process(logger, 'warning',
                f"Expected {task_cfg['num_labels']} labels but found {len(label_map)}"
            )

    # tokenize
    prompt_key = task_cfg['prompt_key'] if args.use_task_prompts else None
    if prompt_key:
        log_from_main_process(logger, 'info',
            f"Task prompts enabled: key={prompt_key}")

    log_from_main_process(logger, 'info',"Tokenizing training data...")
    train_examples = tokenize_for_classifier(
        [train_ds[i] for i in range(len(train_ds))],
        tokenizer, args.task, label_map,
        args.max_encoder_length, args.max_decoder_length,
        prompt_key=prompt_key, 
        chat_token_ids=chat_token_ids,
    )
    log_from_main_process(logger, 'info',f"Tokenized {len(train_examples)} training examples")

    val_examples = None
    if val_ds:
        log_from_main_process(logger, 'info',"Tokenizing validation data...")
        val_examples = tokenize_for_classifier(
            [val_ds[i] for i in range(len(val_ds))],
            tokenizer, args.task, label_map,
            args.max_encoder_length, args.max_decoder_length,
            prompt_key=prompt_key,
            chat_token_ids=chat_token_ids,
        )
        log_from_main_process(logger, 'info',f"Tokenized {len(val_examples)} validation examples")

    # initialize model
    model_rngs = nnx.Rngs(params=args.seed, dropout=args.seed + 1)
    char_buckets = np.ones((model_config.vocab_size, 128)) if model_config.char_subwords else None
    jamo_buckets = np.ones((model_config.vocab_size, 128)) if model_config.jamo_subwords else None

    with jax.set_mesh(mesh):
        @nnx.jit
        def init_model(init_rngs):
            m = FlaxHan2HanForSequenceClassification(
                model_config,
                init_rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                gradient_checkpointing=args.remat_policy != "none",
                sharding=parallelism_config['model_param_sharding'],
                char_buckets=char_buckets,
                jamo_buckets=jamo_buckets,
            )
            return m

        model = init_model(model_rngs)

    # restore pretrained weights into model.model (the backbone)
    # auto-detect checkpoint item names (handles both training checkpoints
    # with optimizer states and averaged checkpoints with model+meta only)
    import orbax.checkpoint as ocp
    from etils import epath
    ckpt_path = epath.Path(args.model_path)
    if not str(ckpt_path).endswith('/checkpoints'):
        ckpt_path = ckpt_path / "checkpoints"
    ckpt_steps = sorted(int(p.name) for p in ckpt_path.iterdir() if p.name.isdigit())
    ckpt_items = tuple(
        p.name for p in (ckpt_path / str(ckpt_steps[-1])).iterdir() if p.is_dir()
    )
    log_from_main_process(logger, 'info',f"Detected checkpoint items: {ckpt_items}")
    pretrained_ckpt_manager = ocp.CheckpointManager(
        args.model_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=1, read_only=True),
        item_names=ckpt_items,
    )

    # checkpoint averaging (matching finetune_sft.py logic)
    avg_n = args.average_last_n or 0
    avg_steps = args.average_steps
    avg_weights = args.average_weights
    do_averaging = avg_n > 0 or avg_steps is not None

    backbone = model.model

    if do_averaging:
        all_steps = sorted(pretrained_ckpt_manager.all_steps())
        if not all_steps:
            raise ValueError(f"No checkpoint steps found at {args.model_path}")

        if avg_steps is not None:
            steps_to_avg = sorted(avg_steps)
            missing = [s for s in steps_to_avg if s not in all_steps]
            if missing:
                raise ValueError(f"Checkpoint steps not found: {missing}")
        else:
            steps_to_avg = all_steps[-avg_n:]

        if avg_weights is not None:
            if len(avg_weights) != len(steps_to_avg):
                raise ValueError(
                    f"Got {len(avg_weights)} weights for {len(steps_to_avg)} checkpoints"
                )
            w = jnp.array(avg_weights) / sum(avg_weights)
        else:
            w = jnp.array([1.0 / len(steps_to_avg)] * len(steps_to_avg))

        weights_str = ', '.join(f'{float(x):.3f}' for x in w)
        log_from_main_process(logger, 'info',f"Averaging {len(steps_to_avg)} checkpoints: {steps_to_avg}")
        log_from_main_process(logger, 'info',f"Weights: [{weights_str}]")

        accumulated = None
        for i, step in enumerate(steps_to_avg):
            log_from_main_process(logger, 'info',f"Loading checkpoint step {step} ({i+1}/{len(steps_to_avg)})...")
            restore_checkpoint(
                pretrained_ckpt_manager,
                backbone,
                optimizers=None,
                step=step,
                mesh=mesh,
                use_abstract_restoration=True,
                model_only=True,
            )
            state = nnx.state(backbone, nnx.Param)
            wi = float(w[i])
            if accumulated is None:
                accumulated = jax.tree.map(
                    lambda x: x.astype(jnp.float32) * wi, state
                )
            else:
                accumulated = jax.tree.map(
                    lambda acc, x: acc + x.astype(jnp.float32) * wi,
                    accumulated, state,
                )

        param_state = nnx.state(backbone, nnx.Param)
        final_state = jax.tree.map(
            lambda acc, orig: acc.astype(orig.dtype),
            accumulated, param_state,
        )
        nnx.update(backbone, final_state)
        log_from_main_process(logger, 'info',"Checkpoint averaging complete")
        del accumulated, final_state
    else:
        pretrained_step = pretrained_ckpt_manager.latest_step()
        if pretrained_step is None:
            raise ValueError(f"No checkpoint found at {args.model_path}")

        log_from_main_process(logger, 'info',f"Restoring pretrained backbone from step {pretrained_step}")
        restore_checkpoint(
            pretrained_ckpt_manager,
            backbone,
            optimizers=None,
            step=pretrained_step,
            mesh=mesh,
            use_abstract_restoration=True,
            model_only=True,
        )

    log_from_main_process(logger, 'info',"Pretrained backbone restored (classifier head is randomly initialized)")
    del pretrained_ckpt_manager

    # training setup. with gradient_accumulation_steps=k, optimizer.update() is
    # called once per micro-batch but MultiSteps (wrapped inside create_optimizer
    # via _create_gradaccum_schedule) only applies an update every k calls.
    # ``micro_step`` counts loop iterations; ``global_step`` counts completed
    # optimizer updates. progress for the LR schedule is fed on the micro-step
    # axis so warmup_ratio/cooldown_ratio resolve to the same wall-clock horizon
    # regardless of grad_accum.
    micro_steps_per_epoch = len(train_examples) // global_batch_size
    total_micro_steps = micro_steps_per_epoch * args.num_epochs
    gradient_steps_per_epoch = micro_steps_per_epoch // grad_accum
    total_gradient_steps = total_micro_steps // grad_accum
    if total_gradient_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps={grad_accum} exceeds available "
            f"micro-batches per run ({total_micro_steps}). Lower grad_accum, "
            f"increase num_epochs, or shrink batch_size."
        )
    args.total_steps = total_gradient_steps

    log_from_main_process(logger, 'info',
        f"Micro-steps per epoch: {micro_steps_per_epoch}, "
        f"gradient steps per epoch: {gradient_steps_per_epoch}")
    log_from_main_process(logger, 'info',
        f"Total micro-steps: {total_micro_steps}, "
        f"total gradient steps: {total_gradient_steps}, "
        f"warmup_ratio: {args.warmup_ratio} "
        f"({int(total_gradient_steps * args.warmup_ratio)} gradient steps)")

    # parameter freezing
    if args.freeze_backbone:
        wrt_filter = nnx.filterlib.All(
            nnx.Param,
            nnx.filterlib.PathContains('classifier'),
        )
        log_from_main_process(logger, 'info',"Backbone frozen -- only training classifier head")
    else:
        wrt_filter = nnx.Param
        log_from_main_process(logger, 'info',"Full model fine-tuning (backbone + classifier)")

    # progress-based LR schedule + full optimizer chain shared with pretraining /
    # SFT. ``args.max_tokens`` is set from total_micro_steps so any args dependent
    # on it (adaptive_wd warmup, _create_gradaccum_schedule fallbacks) interpret a
    # single micro-step as a single "token" -- progress for the schedule is driven
    # explicitly via the optimizer.update(progress=...) kwarg below. create_optimizer
    # wraps the base chain in optax.MultiSteps(every_k=grad_accum), so the optimizer
    # is updated on every micro-step but only steps the params every grad_accum-th.
    args.max_tokens = max(total_micro_steps, 1)
    lr_schedule = create_learning_rate_schedule(args)
    with mesh:
        optimizer = create_optimizer(args, lr_schedule, model, wrt_filter=wrt_filter)

    # training loop
    best_metric = -float('inf')
    rng = jax.random.PRNGKey(args.seed)
    micro_step = 0
    global_step = 0

    log_from_main_process(logger, 'info', "Entering the training loop...")

    for epoch in range(args.num_epochs):
        epoch_start = time.time()

        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, len(train_examples))
        shuffled = [train_examples[int(i)] for i in perm]

        epoch_losses = []
        for step_in_epoch, batch in enumerate(
            create_batches_from_examples(shuffled, global_batch_size)
        ):
            rng, dropout_rng = jax.random.split(rng)
            dropout_rngs = nnx.Rngs(
                dropout=dropout_rng,
                layerdrop=jax.random.fold_in(dropout_rng, 1),
                default=jax.random.fold_in(dropout_rng, 2),
            )

            progress = jnp.asarray(micro_step / max(total_micro_steps, 1), dtype=jnp.float32)
            with mesh:
                batch_sharded = shard_batch_to_devices(batch, mesh)
                loss = classification_train_step(
                    model, optimizer, batch_sharded,
                    task_cfg['task_type'], dropout_rngs,
                    progress,
                )
            micro_step += 1
            if micro_step % grad_accum == 0:
                global_step += 1

            loss_val = float(loss)
            epoch_losses.append(loss_val)

            if step_in_epoch % 50 == 0:
                log_from_main_process(logger, 'info',
                    f"Epoch {epoch+1}/{args.num_epochs} "
                    f"micro {step_in_epoch}/{micro_steps_per_epoch} "
                    f"(cum {micro_step}/{total_micro_steps}, "
                    f"grad {global_step}/{total_gradient_steps}) "
                    f"loss={loss_val:.4f}"
                )

        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        elapsed = time.time() - epoch_start

        log_msg = f"Epoch {epoch+1}/{args.num_epochs} done in {elapsed:.1f}s, avg loss={avg_loss:.4f}"

        if val_examples:
            model.model.eval()
            metrics = evaluate(
                model, val_examples, args.eval_batch_size,
                task_cfg['task_type'], mesh,
            )
            model.model.train()

            primary = metrics[task_cfg['primary_metric']]
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            log_msg += f" | eval: {metrics_str}"

            if primary > best_metric:
                best_metric = primary
                log_msg += " (new best)"

        log_from_main_process(logger, 'info',log_msg)

    log_from_main_process(logger, 'info',f"Training complete. Best {task_cfg['primary_metric']}: {best_metric:.4f}")


if __name__ == "__main__":
    main()
