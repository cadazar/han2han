#!/usr/bin/env python3
# coding: utf-8

import logging
import sys
# logging setup before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout,
    force=True
)
# suppress absl logging
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.WARNING)

from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List

logger = logging.getLogger(__name__)

class Han2HanConfig(PretrainedConfig):
    model_type = "han2han"
    keys_to_ignore_at_inference = ["past_key_values"]
    has_no_defaults_at_init = True
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "decoder_nlayer",  # HF expects this for decoder
        "num_layers": "decoder_nlayer",
        "encoder_layers": "encoder_nlayer",
        "decoder_layers": "decoder_nlayer",
        "d_kv": "d_prime",
        "intermediate_size": "d_ff",
        "hidden_dropout_prob": "resid_pdrop",
        "attention_probs_dropout_prob": "attn_pdrop",
    }

    # Fields that should always be saved, even if they match PretrainedConfig defaults
    _always_save = ["tie_word_embeddings", "tie_encoder_decoder"]

    # Explicitly list attributes that should always be saved
    def to_diff_dict(self):
        """
        Override to ensure certain fields are always saved, even if they match
        PretrainedConfig defaults.
        """
        # Get the diff dict from parent
        output = super().to_diff_dict()
        
        # Force inclusion of fields we always want to save
        for field in self._always_save:
            if hasattr(self, field):
                output[field] = getattr(self, field)

        return output

    def __init__(
        self,

        jamo_subwords: bool = False,
        char_subwords: bool = False,
        use_han2han_transcription: str = 'false',  # 'false', 'true', 'hangul_only', 'reverse'

        vocab_size: int = 38000,
        jamo_vocab_size: int = 8000,
        char_vocab_size: int = 8000,
        subword_embed_dim: Optional[int] = None,  # hidden dim of wje/wce subtoken embeddings; None defaults to d_model // 2
        char_is_unified_cjk: bool = False,
        decoder_nlayer: int = 4,
        encoder_nlayer: int = 4,
        n_positions: int = 1024,
        d_model: int = 768,
        d_prime: Optional[int] = None,
        rope_theta: float = 10000.0,
        rope_theta_sliding: Optional[float] = None,  # per-layer theta override for 'sliding'/'local' attention layers (None = inherit rope_theta)
        apply_legacy_rope_quirk: Optional[bool] = None,  # None = auto-detect legacy scan_rope_theta quirk; False = opt out (use rope_theta/rope_theta_sliding as specified, for V2+); True = force-apply
        d_ff: int = 3072,
        attention_mechanism: str = 'mha',          # global default attention type for all layers
        encoder_attention_types: Optional[List[str]] = None,  # per-layer encoder self-attention types
        decoder_attention_types: Optional[List[str]] = None,  # per-layer decoder self-attention types
        decoder_cross_attention_types: Optional[List[str]] = None,  # per-layer decoder cross-attention types
        sliding_window_size: int = 256,             # window size for 'mha-sliding' layers (0 = full attention)
        ffn_activation: str = "swiglu",     # 'swiglu', 'geglu', 'reglu2', 'gelu', 'gelu_new', 'relu2'
        dense_ffn_activation: Optional[str] = None,  # activation override; None = follow ffn_activation
        use_fla_fused_mlp: bool = False,
        use_fla_fused_norm: bool = False,
        use_fla_fused_rotary: bool = False,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,                       # MHA only: per-head dim. If both set with d_prime, must be consistent. Required for GQA.
        num_kv_heads: Optional[int] = None,                   # MHA only: KV heads for self-attn (None = num_heads = full MHA). 1 = MQA. Must divide num_heads.
        cross_attn_num_heads: Optional[int] = None,           # MHA only: Q heads for cross-attn (None = num_heads). Lets cross-attn use full d_model fidelity (cross_attn_num_heads * head_dim) while self-attn stays compressed.
        cross_attn_num_kv_heads: Optional[int] = None,        # MHA only: KV heads for cross-attn (None = num_kv_heads). Must divide cross_attn_num_heads.
        use_qk_norm: bool = False,                            # MHA only: per-head RMSNorm on Q and K (Gemma 3 / T5Gemma 2 style)
        query_pre_attn_scalar: Optional[float] = None,        # MHA only: HF-Gemma 3 semantics. Q multiplier = scalar ** -0.5. None = head_dim ** -0.5.
        num_labels: int = 3,
        use_learned_bidirectional: float = True,
        remat_policy: str = "full",
        layer_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        cross_attn_pdrop: float = 0.15,
        classf_pdrop: float = 0.1,
        classifier_head_type: str = 'linear',  # 'linear' (T5Gemma 2) or 'mlp' (RoBERTa-style tanh)
        embedding_dropout_rate: float = 0.0,  # Probability of dropping each type of embedding module (wte/wje/wce)
        layer_norm_epsilon: float = 1e-5,
        decoder_norm_type: str = 'rmsnorm',  # 'rmsnorm', 'rmsnorm_bias', 'layernorm'
        encoder_norm_type: str = 'rmsnorm',  # 'rmsnorm', 'rmsnorm_bias', 'layernorm'
        initializer_range: float = 0.02,
        kernel_init_type: str = 'normal',      # 'normal' or 'variance_scaling'
        kernel_init_scale: float = 0.1,        # scale for variance_scaling (0.1=Switch-style, 1.0=lecun)
        init_biases_normal: bool = False,      # if True, init biases as normal(stddev=initializer_range) (V1 behavior); else zeros
        init_cache: bool = False,
        pad_token_id: int = 1,
        decoder_start_token_id: int = 0,
        tie_word_embeddings: bool = True,
        tie_encoder_decoder: bool = False,
        tie_input_output_embeddings: bool = False,
        tie_subtoken_embeddings: bool = True,
        return_dict: bool = True,
        seed: int = 0,
        eos_token_id: int = 2,
        bos_token_id: int = 0,
        use_bart_training: bool = True,
        use_bart_collator: bool = True,

        # SFT-only token ids. Read by ChatSFTCollator; not used by pretraining.
        # Decoder is primed with <|think|> when reasoning is enabled, otherwise
        # with <|assistant|>; turn closes with <|end_of_turn|>. None means look
        # up from tokenizer at collator init time (recommended).
        sft_decoder_start_token_id_thinking: Optional[int] = None,
        sft_decoder_start_token_id_default: Optional[int] = None,
        sft_eos_token_id: Optional[int] = None,

        use_sub_ln: bool = False,                   # SubLN: RMSNorm before output projections in attn and FFN

        # Bias configuration
        use_bias: bool = True,                      # global toggle for biases in all linear layers

        label_smoothing: float = 0.1,               # label smoothing alpha for cross-entropy loss

        # scan layers (compile one layer body, repeat N times via jax.lax.scan)
        use_scan_layers: bool = False,

        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.jamo_vocab_size = jamo_vocab_size
        self.char_vocab_size = char_vocab_size
        # subtoken embedding hidden dim: defaults to d_model // 2 when unspecified
        self.subword_embed_dim = subword_embed_dim if subword_embed_dim is not None else d_model // 2
        self.char_is_unified_cjk = char_is_unified_cjk
        self.jamo_subwords = jamo_subwords
        self.char_subwords = char_subwords
        self.use_han2han_transcription = use_han2han_transcription
        self.decoder_nlayer = decoder_nlayer
        self.encoder_nlayer = encoder_nlayer
        self.n_positions = n_positions
        self.d_model = d_model
        self.rope_theta = rope_theta
        self.rope_theta_sliding = rope_theta_sliding
        self.apply_legacy_rope_quirk = apply_legacy_rope_quirk
        self.d_prime = d_prime
        self.d_ff = d_ff
        self.ffn_activation = ffn_activation
        self.dense_ffn_activation = dense_ffn_activation if dense_ffn_activation is not None else ffn_activation
        # Only set num_heads for MHA attention
        self.num_heads = num_heads if attention_mechanism == 'mha' or any(
            'mha' in attn_type for attn_type in (encoder_attention_types or [])) else None
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.cross_attn_num_heads = cross_attn_num_heads
        self.cross_attn_num_kv_heads = cross_attn_num_kv_heads
        self.use_qk_norm = use_qk_norm
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.use_fla_fused_mlp = use_fla_fused_mlp
        self.use_fla_fused_norm = use_fla_fused_norm
        self.use_fla_fused_rotary = use_fla_fused_rotary
        self.remat_policy = remat_policy
        self.layer_pdrop = layer_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.cross_attn_pdrop = cross_attn_pdrop
        self.classf_pdrop = classf_pdrop
        self.classifier_head_type = classifier_head_type
        self.embedding_dropout_rate = embedding_dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.decoder_norm_type = decoder_norm_type
        self.encoder_norm_type = encoder_norm_type
        self.initializer_range = initializer_range
        self.kernel_init_type = kernel_init_type
        self.kernel_init_scale = kernel_init_scale
        self.init_biases_normal = init_biases_normal
        self.use_cache = init_cache
        self.tie_input_output_embeddings = tie_input_output_embeddings
        self.use_sub_ln = use_sub_ln

        # bias configuration
        self.use_bias = use_bias

        self.label_smoothing = label_smoothing
        self.use_scan_layers = use_scan_layers

        super().__init__(
            pad_token_id = pad_token_id,
            decoder_start_token_id = decoder_start_token_id,
            tie_encoder_decoder = tie_encoder_decoder,
            tie_word_embeddings = tie_word_embeddings,
            bos_token_id = bos_token_id,
            eos_token_id = eos_token_id,
            **kwargs
        )

        self.num_labels = num_labels
        self.is_encoder_decoder = True
        self.tie_word_embeddings = tie_word_embeddings
        self.tie_encoder_decoder = tie_encoder_decoder
        self.tie_subtoken_embeddings = tie_subtoken_embeddings
        self.seed = seed
        self.return_dict = return_dict
        self.eos_token_id = eos_token_id
        self.sft_decoder_start_token_id_thinking = sft_decoder_start_token_id_thinking
        self.sft_decoder_start_token_id_default = sft_decoder_start_token_id_default
        self.sft_eos_token_id = sft_eos_token_id
        self.attention_mechanism = attention_mechanism
        self.encoder_attention_types = encoder_attention_types
        self.decoder_attention_types = decoder_attention_types
        self.decoder_cross_attention_types = decoder_cross_attention_types
        self.sliding_window_size = sliding_window_size
        self.use_learned_bidirectional = use_learned_bidirectional
        self.use_bart_training = use_bart_training
        self.use_bart_collator = use_bart_collator

        self._sanitize_d_prime(encoder_attention_types, decoder_attention_types, decoder_cross_attention_types)
        self._resolve_mha_attention_fields(encoder_attention_types, decoder_attention_types, decoder_cross_attention_types)

        if self.d_prime > 256 and self.use_fla_fused_rotary:
            logger.warning(
                "WARNING: `d_prime` is greater than 256 and `use_fla_fused_rotary` is set to `True`. "
                "This is not supported. Falling back to non-fused rotary embeddings."
            )
            self.use_fla_fused_rotary = False

        self._validate_attention_config()

        if self.use_scan_layers:
            self._validate_scan_config()

        self._apply_legacy_rope_quirk()

    def _apply_legacy_rope_quirk(self) -> None:
        """Rewrite rope_theta when the legacy Flax `_scan_key` quirk applied.

        Pre-fix Flax `_identify_scan_groups` collapsed all MHA variants
        ('mha', 'mha-sliding', 'mha-local') under a single scan key. So when
        `use_scan_layers=True` and the attention pattern mixes 'mha' with
        'mha-sliding' or 'mha-local', all layers in the resulting scan stack
        share one `FlaxHan2HanBlock` instance whose `effective_rope_theta` is
        baked from `position_specs[0]`. In practice the configured
        `rope_theta` field was overwritten by `rope_theta_sliding` whenever
        the first attention layer was sliding.

        Checkpoints trained under that quirk have a saved `rope_theta` that
        doesn't reflect what the weights actually saw. This sanitizer detects
        the pattern and rewrites the in-memory config so the runtime loads
        the model with values matching its training behavior. Flax post-fix
        resolves `effective_rope_theta` per call from `override_window_size`,
        so the sanitizer's rewrite is what makes legacy checkpoints behave
        identically under the corrected dispatch path.

        Honors `self.apply_legacy_rope_quirk`:
          - None (default): auto-apply when the heuristic triggers (existing
            checkpoints, no opt-out specified -- safe default).
          - False: explicit opt-out (V2+ runs that want hybrid rope_theta).
            Sanitizer becomes a no-op even when the trigger pattern matches.
          - True: force-apply (mostly for testing).
        """
        opt = getattr(self, 'apply_legacy_rope_quirk', None)
        if opt is False:
            return

        if not getattr(self, 'use_scan_layers', False):
            return
        if getattr(self, 'rope_theta_sliding', None) is None:
            return
        if self.rope_theta == self.rope_theta_sliding:
            return

        def _has_mixed_mha(types):
            if not types:
                return False
            has_sliding = any(
                ('sliding' in t) or ('local' in t)
                for t in types if t
            )
            has_full_mha = any(t == 'mha' for t in types)
            return has_sliding and has_full_mha

        triggered = opt is True or (
            _has_mixed_mha(self.encoder_attention_types)
            or _has_mixed_mha(self.decoder_attention_types)
        )
        if not triggered:
            return

        logger.warning(
            "[Han2HanConfig] Detected legacy scan_rope_theta quirk "
            "(use_scan_layers=True + mixed mha/mha-sliding pattern + "
            "rope_theta != rope_theta_sliding). Pre-fix Flax "
            "_identify_scan_groups collapsed MHA variants into one scan "
            "stack whose rope_theta was baked from position_specs[0], so "
            "the saved rope_theta=%s was overwritten by rope_theta_sliding=%s "
            "during training. Rewriting in-memory rope_theta=%s to match "
            "actual training behavior. To opt out for new training, set "
            "apply_legacy_rope_quirk=False (CLI: --no_apply_legacy_rope_quirk).",
            self.rope_theta, self.rope_theta_sliding, self.rope_theta_sliding,
        )
        self.rope_theta = self.rope_theta_sliding

    def _sanitize_d_prime(
        self,
        encoder_attention_types: Optional[List[str]],
        decoder_attention_types: Optional[List[str]],
        decoder_cross_attention_types: Optional[List[str]],
    ) -> None:
        """Sanitize d_prime based on attention configuration.

        - Layerwise config: d_prime defaults to d_model if not specified
        - Global MHA: d_prime defaults to d_model if not specified
        """
        has_layerwise = (
            encoder_attention_types is not None or
            decoder_attention_types is not None or
            decoder_cross_attention_types is not None
        )

        if has_layerwise:
            if self.d_prime is None and self.head_dim is None:
                self.d_prime = self.d_model
                logger.info(f"d_prime not specified for layerwise config, defaulting to d_model={self.d_model}")
        elif self.attention_mechanism == 'mha':
            if self.d_prime is None and self.head_dim is None:
                self.d_prime = self.d_model
                logger.info(f"d_prime not specified for MHA, defaulting to d_model={self.d_model}")
        else:
            if self.d_prime is None:
                raise ValueError(
                    f"d_prime must be explicitly set for attention_mechanism='{self.attention_mechanism}'."
                )

    def _resolve_mha_attention_fields(
        self,
        encoder_attention_types: Optional[List[str]],
        decoder_attention_types: Optional[List[str]],
        decoder_cross_attention_types: Optional[List[str]],
    ) -> None:
        """Resolve head_dim/d_prime consistency and num_kv_heads defaults for MHA."""
        all_types = []
        if encoder_attention_types is not None:
            all_types.extend(encoder_attention_types)
        if decoder_attention_types is not None:
            all_types.extend(decoder_attention_types)
        if decoder_cross_attention_types is not None:
            all_types.extend(t for t in decoder_cross_attention_types if t is not None)
        has_layerwise = bool(all_types)
        has_mha = (
            any(t == 'mha' or t.startswith('mha-') for t in all_types)
            if has_layerwise
            else self.attention_mechanism == 'mha'
        )

        if not has_mha:
            return

        if self.num_heads is None:
            raise ValueError(
                "num_heads must be specified for MHA attention."
            )

        if self.head_dim is not None and self.d_prime is not None:
            if self.head_dim * self.num_heads != self.d_prime:
                raise ValueError(
                    f"head_dim ({self.head_dim}) * num_heads ({self.num_heads}) = "
                    f"{self.head_dim * self.num_heads} does not match d_prime ({self.d_prime})."
                )
        elif self.head_dim is not None:
            self.d_prime = self.head_dim * self.num_heads
        elif self.d_prime is not None:
            if self.d_prime % self.num_heads != 0:
                raise ValueError(
                    f"d_prime ({self.d_prime}) must be divisible by num_heads ({self.num_heads})."
                )
            self.head_dim = self.d_prime // self.num_heads
        else:
            raise ValueError(
                "MHA requires at least one of head_dim or d_prime to be set."
            )

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.num_kv_heads < 1 or self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be >= 1 and divide "
                f"num_heads ({self.num_heads}) evenly."
            )

        if self.cross_attn_num_heads is None:
            self.cross_attn_num_heads = self.num_heads
        if self.cross_attn_num_heads < 1:
            raise ValueError(
                f"cross_attn_num_heads ({self.cross_attn_num_heads}) must be >= 1."
            )

        if self.cross_attn_num_kv_heads is None:
            self.cross_attn_num_kv_heads = self.num_kv_heads
        if self.cross_attn_num_kv_heads < 1 or self.cross_attn_num_heads % self.cross_attn_num_kv_heads != 0:
            raise ValueError(
                f"cross_attn_num_kv_heads ({self.cross_attn_num_kv_heads}) must be >= 1 and divide "
                f"cross_attn_num_heads ({self.cross_attn_num_heads}) evenly."
            )

        if self.query_pre_attn_scalar is not None and self.query_pre_attn_scalar <= 0:
            raise ValueError(
                f"query_pre_attn_scalar must be positive, got {self.query_pre_attn_scalar}."
            )

    def _validate_attention_config(self) -> None:
        """Validate mutual exclusivity of global vs layerwise attention config."""
        has_layerwise = (
            self.encoder_attention_types is not None or
            self.decoder_attention_types is not None or
            self.decoder_cross_attention_types is not None
        )

        if has_layerwise:
            missing = []
            if self.encoder_attention_types is None:
                missing.append('encoder_attention_types')
            if self.decoder_attention_types is None:
                missing.append('decoder_attention_types')
            if self.decoder_cross_attention_types is None:
                missing.append('decoder_cross_attention_types')

            if missing:
                raise ValueError(
                    f"Per-layer attention mode requires all three lists to be specified. "
                    f"Missing: {missing}. Either specify all layerwise types, or remove "
                    f"all and use 'attention_mechanism' for a global default."
                )

            if self.attention_mechanism is not None:
                logger.warning(
                    f"Both attention_mechanism ('{self.attention_mechanism}') and layerwise "
                    f"attention types are set. Layerwise types take precedence; "
                    f"attention_mechanism will be ignored."
                )
        else:
            if self.attention_mechanism is None:
                raise ValueError(
                    "No attention configuration specified. Either set 'attention_mechanism' "
                    "for a global default, or specify all three layerwise types: "
                    "encoder_attention_types, decoder_attention_types, decoder_cross_attention_types."
                )

    def _validate_scan_config(self) -> None:
        """Validate config constraints for scanned layer execution."""
        # layerdrop breaks scan (stochastic layer skip)
        if self.layer_pdrop > 0:
            raise ValueError(
                f"use_scan_layers is incompatible with layerdrop > 0. "
                f"Got layer_pdrop={self.layer_pdrop}. Set layerdrop: 0.0."
            )

    def make_kernel_init(self, dtype=None):
        """Create kernel initializer based on config."""
        from flax import nnx
        kwargs = {'dtype': dtype} if dtype is not None else {}
        if self.kernel_init_type == 'variance_scaling':
            return nnx.initializers.variance_scaling(
                self.kernel_init_scale, 'fan_in', 'truncated_normal', **kwargs
            )
        return nnx.initializers.normal(stddev=self.initializer_range, **kwargs)

    def make_bias_init(self):
        """Create bias initializer based on config.

        V1 behavior is normal(stddev=initializer_range); current default is zeros.
        """
        from flax import nnx
        if self.init_biases_normal:
            return nnx.initializers.normal(stddev=self.initializer_range)
        return nnx.initializers.zeros_init()

    def get_encoder_attention_types(self) -> List[str]:
        """Get expanded per-layer encoder attention types.

        If encoder_attention_types is None, returns [attention_mechanism] * encoder_nlayer.
        If specified, expands short patterns by repetition (pattern length must divide layer count).
        """
        return self._expand_attention_types(
            self.encoder_attention_types,
            self.encoder_nlayer,
            "encoder_attention_types"
        )

    def get_decoder_attention_types(self) -> List[str]:
        """Get expanded per-layer decoder self-attention types."""
        return self._expand_attention_types(
            self.decoder_attention_types,
            self.decoder_nlayer,
            "decoder_attention_types"
        )

    def get_decoder_cross_attention_types(self) -> List[str]:
        """Get expanded per-layer decoder cross-attention types."""
        return self._expand_attention_types(
            self.decoder_cross_attention_types,
            self.decoder_nlayer,
            "decoder_cross_attention_types"
        )

    def _expand_attention_types(
        self,
        attention_types: Optional[List[str]],
        num_layers: int,
        field_name: str
    ) -> List[str]:
        """Expand short attention type patterns to full layer count.

        Args:
            attention_types: List of attention types or None
            num_layers: Number of layers to expand to
            field_name: Name of field for error messages

        Returns:
            List of attention types with length == num_layers
        """
        if attention_types is None:
            if self.attention_mechanism is None:
                raise ValueError(
                    f"{field_name} is None and no global attention_mechanism fallback. "
                    f"This should have been caught by _validate_attention_config()."
                )
            return [self.attention_mechanism] * num_layers

        if len(attention_types) == num_layers:
            return attention_types

        if num_layers % len(attention_types) != 0:
            raise ValueError(
                f"{field_name} length ({len(attention_types)}) must divide "
                f"layer count ({num_layers}) evenly for pattern repetition"
            )

        repeats = num_layers // len(attention_types)
        return attention_types * repeats