import flax
flax.config.update("flax_always_shard_variable", False)
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
REPO = sys.argv[1]
TAG = sys.argv[2]
sys.path.insert(0, REPO)
import jax, jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from flax import nnx
from han2han_config import Han2HanConfig
import modeling_han2han_flax as M

cfg = Han2HanConfig(
    vocab_size=4096, n_positions=2048,
    d_model=640, d_ff=2048, subword_embed_dim=384,
    encoder_nlayer=2, decoder_nlayer=2,
    num_heads=4, num_kv_heads=1, head_dim=256,
    cross_attn_num_heads=4, cross_attn_num_kv_heads=2,
    use_qk_norm=True, query_pre_attn_scalar=256,
    sliding_window_size=128, rope_theta=500000, rope_theta_sliding=10000,
    apply_legacy_rope_quirk=False,
    encoder_attention_types=['mha-sliding', 'mha'],
    decoder_attention_types=['mha-sliding', 'mha'],
    decoder_cross_attention_types=['mha'],
    tie_encoder_decoder=True, tie_word_embeddings=True,
    tie_subtoken_embeddings=False, tie_input_output_embeddings=True,
    ffn_activation='geglu', use_sub_ln=True, use_bias=True,
    jamo_subwords=True, char_subwords=True, char_is_unified_cjk=False,
    use_scan_layers=False, remat_policy='none',
)
rngs = nnx.Rngs(0)
model = M.FlaxHan2Han(cfg, rngs=rngs, dtype=jnp.float32)
params = nnx.state(model, nnx.Param)
n = sum(int(jnp.size(p)) for p in jax.tree_util.tree_leaves(params))
keys = sorted('/'.join(str(k) for k in path) for path, _ in jax.tree_util.tree_leaves_with_path(params))

mesh = Mesh(np.array(jax.devices('cpu')).reshape(1, 1), ('data', 'model'))
B, S = 2, 16
rng = np.random.default_rng(0)
ids = jnp.asarray(rng.integers(0, 4096, size=(B, S)), dtype=jnp.int32)
am = jnp.ones((B, S), jnp.int32)
with jax.set_mesh(mesh):
    out = model(input_ids=ids, attention_mask=am,
                decoder_input_ids=ids, decoder_attention_mask=am)
logits = np.asarray(out.logits, dtype=np.float64)
np.savez(f"/tmp/parity_{TAG}.npz",
         n=n, logits=logits, argmax=np.asarray(out.logits.argmax(-1)),
         keys=np.array(keys))
print(f"[{TAG}] PARAM COUNT: {n}  n_param_tensors: {len(keys)}")
print(f"[{TAG}] logits sum: {logits.sum():.6f}  shape: {logits.shape}")
