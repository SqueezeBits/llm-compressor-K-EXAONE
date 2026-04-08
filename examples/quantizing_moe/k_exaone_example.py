from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.k_exaone_moe import CalibrationExaoneMoeSparseMoEBlock  # noqa: F401
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import AWQMapping

# Select model.
# K-EXAONE-236B-A23B is a Mixture-of-Experts model with 128 routed experts
# and 1 shared expert per MoE layer (config: num_shared_experts=1).
#
# Note: requires transformers >= 5.1.0 (exaone_moe model type was added in 5.

model_id = "LGAI-EXAONE/K-EXAONE-236B-A23B"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# MoE calibration is now handled automatically by the pipeline.
# The `CalibrationExaoneMoeSparseMoEBlock` module (from
# `llmcompressor.modeling.k_exaone_moe`) will be applied during calibration
# to enable proper expert calibration.
#
# It replaces `ExaoneMoeSparseMoEBlock` and converts the batched
# `ExaoneMoeExperts` 3D weight tensors (gate_up_proj, down_proj) into a
# ModuleList of individual nn.Linear modules — one per expert — so that AWQ
# observers can attach to every expert's projections.
#
# `is_permanent = True`: the replacement is kept for the full quantization run
# (not restored after calibration), because AWQ compresses the individual
# nn.Linear modules in-place.

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# For K-EXAONE-236B MoE: AWQ accumulates raw input activations for every
# hooked nn.Linear (256 modules per sparse layer: 128 experts × gate+up).
# Storage per module ≈ N_samples × seq_len × 6144 × 2 bytes.
# With device_map="cpu" the model already occupies ~472 GB RAM, leaving
# ~40 GB for activations. Budget:
#   32 samples × 512 tokens × 6144 × 2B × 256 modules ≈ 50 GB — fits.
#   128 samples × 2048 tokens × same                  ≈ 820 GB — OOM.
NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Derive dense-layer indices from config so the ignore list stays correct
# across ExaoneMoe variants (e.g. models with more than one dense prefix layer).
# config.mlp_layer_types is a per-layer list: "dense" | "sparse".
# config.first_k_dense_replace is an integer alternative; mlp_layer_types is
# more explicit and preferred.
dense_mlp_ignores = [
    f"model.layers.{i}.mlp.*"
    for i, layer_type in enumerate(model.config.mlp_layer_types)
    if layer_type == "dense"
]

# Note on MTP (multi-token prediction) layers:
# K-EXAONE includes num_nextn_predict_layers=1 MTP weights in its checkpoint,
# but AutoModelForCausalLM excludes them via _keys_to_ignore_on_load_unexpected.
# They are therefore not present as model submodules and require no explicit
# ignore entry here. If you load a model that does include MTP modules, add
# "re:.*nextn.*" to the ignore list below.

# Build per-sparse-layer post_attention_layernorm → expert mappings.
#
# WHY not use the auto-inferred _exaone_moe_mappings regex?
# K-EXAONE layer 0 is dense: it has post_attention_layernorm but no
# mlp.experts.* or mlp.shared_experts.*. match_modules_set groups by
# lowest-common-ancestor context; because layer 0 never satisfies the
# expert balance targets, parent_context rises to "model.layers" and
# never resets — all 48 post_attention_layernorm modules end up in one
# set, triggering "AWQ needs to match a single smoothlayer" ValueError.
#
# Explicit per-layer exact-name smooth targets keep each set to exactly
# one post_attention_layernorm. Dense layer 0 is omitted entirely (its
# MLP is already excluded from quantization via dense_mlp_ignores).
sparse_layer_indices = [
    i for i, lt in enumerate(model.config.mlp_layer_types) if lt == "sparse"
]
moe_smooth_mappings = [
    AWQMapping(
        f"model.layers.{i}.post_attention_layernorm",
        [
            f"re:model\\.layers\\.{i}\\.mlp\\.experts\\.[0-9]+\\.gate_proj$",
            f"re:model\\.layers\\.{i}\\.mlp\\.experts\\.[0-9]+\\.up_proj$",
            f"model.layers.{i}.mlp.shared_experts.gate_proj",
            f"model.layers.{i}.mlp.shared_experts.up_proj",
        ],
    )
    for i in sparse_layer_indices
]

# Configure the quantization algorithm to run.
# Ignore list:
#   lm_head          — output projection, kept at full precision
#   re:.*gate.weight — MoE router weights (sigmoid-gated, sensitive to quant)
#   dense_mlp_ignores — dense MLP layers derived from config.mlp_layer_types
recipe = AWQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head", "re:.*gate.weight", *dense_mlp_ignores],
    mappings=[
        # v_proj → o_proj: skipped at runtime due to GQA shape incompatibility
        # (o_proj input dim 8192 ≠ v_proj output dim 1024), but kept here so
        # smoothing is applied on models where it does match.
        AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
        # MoE layers: one mapping per sparse layer (avoids regex grouping bug)
        *moe_smooth_mappings,
        AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
    ],
)

# Apply algorithms.
# sequential_targets keeps only one decoder layer in GPU memory at a time,
# which is essential for the 236B model — especially with calibrate_all_experts
# enabled, where all 128 experts run on every calibration token.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["ExaoneMoeDecoderLayer"],
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
