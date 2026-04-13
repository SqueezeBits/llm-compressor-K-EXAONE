from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.k_exaone_moe import CalibrationExaoneMoeSparseMoEBlock  # noqa: F401
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import AWQMapping

# Select model.
# K-EXAONE-236B-A23B is a Mixture-of-Experts model with 128 routed experts and 1 shared expert per MoE layer (config: num_shared_experts=1).

# Note: requires transformers >= 5.1.0

model_id = "LGAI-EXAONE/K-EXAONE-236B-A23B"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# MoE calibration is now handled automatically by the pipeline.

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

NUM_CALIBRATION_SAMPLES = 512
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

dense_mlp_ignores = [
    f"model.layers.{i}.mlp.*"
    for i, layer_type in enumerate(model.config.mlp_layer_types)
    if layer_type == "dense"
]

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
        # MoE layers: one mapping per sparse layer (avoids cross-layer LCA bug)
        *moe_smooth_mappings,
        AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
    ],
    cache_chunk_size_batches=32,
)

# Apply algorithms
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["ExaoneMoeDecoderLayer"],
)

model.config.architectures = ["ExaoneMoEForCausalLM"]

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
