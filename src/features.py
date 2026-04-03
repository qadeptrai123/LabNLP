"""
Stylometric feature extractor — AgnosticFeatureExtractor.
Used by preprocess.py to build agnostic_features columns.
Fixed: NaN/Inf guard on perplexity computation.
"""
import re, math
import numpy as np
import torch
from collections import Counter
from typing import List, Dict

from src.config import MULTI_LANG_KEYWORDS


FEATURE_NAMES = [
    "perplexity",        # 0  (actually cross-entropy loss)
    "id_len_avg",        # 1
    "id_entropy",        # 2
    "id_short_ratio",    # 3
    "id_num_ratio",      # 4
    "style_consistency", # 5
    "spacing_ratio",     # 6
    "line_len_std",      # 7
    "ttr",               # 8
    "comment_ratio",     # 9
    "human_markers",     # 10
]
FEATURE_COUNT = len(FEATURE_NAMES)


class AgnosticFeatureExtractor:
    """
    Extracts language-agnostic stylometric features from source code.

    Feature vector (11 dims):
      [0] perplexity         — cross-entropy loss from Qwen2.5-Coder-1.5B-Instruct
      [1] id_len_avg         — mean identifier length
      [2] id_entropy         — character entropy of all identifiers
      [3] id_short_ratio     — ratio of identifiers <= 2 chars
      [4] id_num_ratio       — ratio of identifiers containing digits
      [5] style_consistency  — Camel vs Snake bias (1.0 = pure one style)
      [6] spacing_ratio     — ratio of '=' without surrounding spaces
      [7] line_len_std      — std of non-empty line lengths
      [8] ttr               — type-token ratio (vocabulary richness)
      [9] comment_ratio      — fraction of non-empty lines that are comments
     [10] human_markers      — TODO/FIXME/XXX/HACK/DEBUG present
    """

    def __init__(self, model_path: str, device: str, max_len: int = 512):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.max_len = max_len

        print(f"Loading Perplexity Model from {model_path} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
            trust_remote_code=True,
        ).eval()
        print("Perplexity model ready.")

        # Compile regexes once
        self.re_words   = re.compile(r'\w+')
        self.re_camel   = re.compile(r'[a-z][A-Z]')
        self.re_snake   = re.compile(r'_')
        self.re_digits  = re.compile(r'\d')
        self.re_eq_sp   = re.compile(r' = ')
        self.re_eq_no   = re.compile(r'(?<=[^\s])=(?=[^\s])')

    def get_feature_names(self) -> List[str]:
        return FEATURE_NAMES

    def _compute_perplexity(self, code: str) -> float:
        """
        Returns cross-entropy loss (NOT perplexity = exp(loss)).
        Guard: returns 0.0 for empty code or NaN/Inf loss (1-token fragments).
        """
        if not code.strip():
            return 0.0
        try:
            inputs = self.tokenizer(
                code, return_tensors="pt", truncation=True, max_length=self.max_len
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=inputs.input_ids, labels=inputs.input_ids)
            loss = outputs.loss.item()
            # Guard: 1-token sequences produce NaN loss
            if math.isnan(loss) or math.isinf(loss):
                return 0.0
            return loss
        except Exception:
            return 0.0

    def compute_perplexity_batch(self, codes: list[str]) -> list[float]:
        """
        Batched perplexity computation — much faster than single-sample.
        Returns cross-entropy loss per sequence. Uses dynamic padding per batch
        to avoid wasting compute on pad tokens.
        """
        valid = [(i, c) for i, c in enumerate(codes) if c.strip()]
        if not valid:
            return [0.0] * len(codes)

        # Separate valid and empty codes
        valid_codes = [c for _, c in valid]
        valid_idx   = [i for i, _ in valid]

        # Tokenize as batch (dynamic padding)
        batch = self.tokenizer(
            valid_codes,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding=True,
        ).to(self.device)

        input_ids  = batch["input_ids"]
        attention  = batch["attention_mask"]

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention,
                labels=input_ids,
            )

        # Per-sequence cross-entropy (ignore pad tokens via attention mask)
        # logits: (batch, seq_len, vocab) → shift right so label[i] = input_ids[i+1]
        logits = outputs.logits[:, :-1, :]          # (B, L-1, V)
        labels = input_ids[:, 1:]                   # (B, L-1)
        mask   = attention[:, 1:].float()           # (B, L-1)

        # Cross-entropy per token
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        nll = nll * mask

        # Per-sequence average (avoid pad inflation)
        seq_losses = nll.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        result = [0.0] * len(codes)
        for local_i, global_i in enumerate(valid_idx):
            v = seq_losses[local_i].item()
            result[global_i] = 0.0 if (math.isnan(v) or math.isinf(v)) else v

        return result

    def _analyze_identifiers(self, words: List[str]) -> List[float]:
        identifiers = [
            w for w in words
            if w not in MULTI_LANG_KEYWORDS and not w.isdigit()
        ]
        if not identifiers:
            return [0.0, 0.0, 0.0, 0.0]

        lens = [len(w) for w in identifiers]
        avg_len = np.mean(lens)

        all_chars = "".join(identifiers)
        if all_chars:
            counts = Counter(all_chars)
            total  = sum(counts.values())
            entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        else:
            entropy = 0.0

        short_ratio = sum(1 for w in identifiers if len(w) <= 2) / len(identifiers)
        num_ratio   = sum(1 for w in identifiers if self.re_digits.search(w)) / len(identifiers)

        return [avg_len, entropy, short_ratio, num_ratio]

    def _analyze_consistency(self, code: str, words: List[str]) -> List[float]:
        identifiers = [w for w in words if w not in MULTI_LANG_KEYWORDS]

        snake = sum(1 for w in identifiers if '_' in w)
        camel = sum(1 for w in identifiers if self.re_camel.search(w))
        total = snake + camel

        consistency = abs(snake - camel) / total if total else 0.0

        spaced  = len(self.re_eq_sp.findall(code))
        nospace = len(self.re_eq_no.findall(code))
        spacing = nospace / (spaced + nospace) if (spaced + nospace) else 0.0

        return [consistency, spacing]

    def _analyze_structure(self, code: str, words: List[str]) -> List[float]:
        lines        = code.split('\n')
        non_empty    = [l for l in lines if l.strip()]
        line_lens    = [len(l) for l in non_empty] if non_empty else [0]
        line_std     = float(np.std(line_lens)) if len(line_lens) > 1 else 0.0
        ttr          = len(set(words)) / len(words) if words else 0.0

        comment_lines = sum(
            1 for l in non_empty if l.strip().startswith(('#', '//', '/*'))
        )
        comment_ratio = comment_lines / (len(non_empty) + 1)

        markers      = len(re.findall(r'\b(TODO|FIXME|XXX|HACK|DEBUG)\b', code, re.I))
        marker_score = 1.0 if markers > 0 else 0.0

        return [line_std, ttr, comment_ratio, marker_score]

    def extract_all(self, code: str) -> List[float]:
        """Main entry point — returns 11-dim feature vector."""
        if not isinstance(code, str):
            code = str(code)

        perp    = self._compute_perplexity(code)
        words   = self.re_words.findall(code)
        f_ids   = self._analyze_identifiers(words)
        f_const = self._analyze_consistency(code, words)
        f_struc = self._analyze_structure(code, words)

        # [1] + [4] + [2] + [4] = 11
        return [perp] + f_ids + f_const + f_struc
