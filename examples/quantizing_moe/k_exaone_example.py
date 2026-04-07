from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.k_exaone_moe import CalibrationExaoneMoeSparseMoEBlock  # noqa: F401
from llmcompressor.modifiers.awq import AWQModifier

# Select model.
# K-EXAONE-236B-A23B is a Mixture-of-Experts model with 128 routed experts
# and 2 shared experts per layer.
#
# Note: requires transformers >= 5.1.0 (exaone_moe model type was added in 5.1)

model_id = "LGAI-EXAONE/K-EXAONE-236B-A23B"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
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

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

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

# Configure the quantization algorithm to run.
# Ignore list:
#   lm_head           — output projection, kept at full precision
#   re:.*gate.weight  — MoE router weights (sigmoid-gated, sensitive to quant)
#   re:.*\.0\.mlp\..* — layer 0 is a dense ExaoneMoeMLP, not a MoE block
recipe = AWQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*gate.weight",
        "re:.*\\.0\\.mlp\\..*",
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
