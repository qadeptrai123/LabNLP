"""
Shared configuration for SemEval Task A.
Single source of truth — imported by both preprocess.py and train.py.
"""
import os

# ── Paths ───────────────────────────────────────────────────────────────────────
_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))   # .../src/
_PROJECT_ROOT   = os.path.dirname(_BASE_DIR)                    # .../LabNLP/
_MODEL_LOCAL_DIR = os.path.join(_PROJECT_ROOT, "model")         # .../LabNLP/model/

# Local model paths (no HF download needed)
UNIXCODER_PATH  = os.path.join(_MODEL_LOCAL_DIR, "unixcoder-base")
QWENTT_PATH     = os.path.join(_MODEL_LOCAL_DIR, "Qwen2.5-Coder-1.5B-Instruct")

DEFAULT_CONFIG = {
    "common": {
        "seed": 42,
        "project_name": "semeval-task13-subtaskA",
    },
    "data": {
        "raw_data_dir":         os.path.join(_PROJECT_ROOT, "Task_A"),
        "data_dir":             os.path.join(_PROJECT_ROOT, "Task_A_Processed"),
        "max_length":           512,
        "calc_perplexity":       True,
        # Local path — perplexity model loaded from disk, not HF
        "perplexity_model_path": QWENTT_PATH,
        "num_handcrafted_features": 11,
    },
    "model": {
        # Local path — UnixCoder loaded from disk, not HF
        "base_model":              UNIXCODER_PATH,
        "num_labels":              2,
        "use_lora":                False,
        "hidden_dropout":           0.2,
        "gradient_checkpointing":   True,
    },
    "training": {
        "batch_size":               32,
        "gradient_accumulation_steps": 2,
        "num_epochs":               10,
        "learning_rate":            2.0e-5,
        "weight_decay":             0.01,
        "focal_gamma":              2.0,
        "use_supcon":               True,
        "checkpoint_dir":           os.path.join(_PROJECT_ROOT, "checkpoints"),
        "early_stop_patience":      4,
        "supcon_weight":            0.1,
    },
}


MULTI_LANG_KEYWORDS = {
    # Control flow
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return',
    # Types
    'int', 'float', 'double', 'char', 'void', 'bool', 'boolean', 'string', 'var', 'let', 'const',
    # Declarations
    'def', 'function', 'class', 'struct', 'interface', 'package', 'import', 'using', 'namespace',
    # Modifiers
    'public', 'private', 'protected', 'static', 'final', 'try', 'catch', 'finally',
    # Operators
    'throw', 'throws', 'new', 'delete', 'true', 'false', 'null', 'nil', 'None', 'self', 'this',
    # Other
    'func', 'defer', 'go', 'map', 'chan', 'type', 'range',
}
