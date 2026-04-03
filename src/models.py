"""
Neural network modules: AttentionPooler, FeatureGatingNetwork, HybridClassifier.
Used by train.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class AttentionPooler(nn.Module):
    """
    Learned attention-weighted pooling over token hidden states.
    Replaces the standard [CLS] token pooling.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense    = nn.Linear(hidden_size, hidden_size)
        self.dropout  = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        # [B, Seq, H]
        x      = torch.tanh(self.dense(hidden_states))
        x      = self.dropout(x)
        scores = self.out_proj(x).squeeze(-1)            # [B, Seq]
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B, Seq, 1]
        return torch.sum(hidden_states * weights, dim=1)   # [B, H]


class FeatureGatingNetwork(nn.Module):
    """
    Processes the 11 stylometric features into a 128-dim embedding.
    This embedding is concatenated with the text embedding for classification.
    """

    def __init__(self, input_dim: int = 11, output_dim: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class HybridClassifier(nn.Module):
    """
    Two-pathway classifier:
      A. Semantic path  — UnixCoder text backbone + AttentionPooler
      B. Agnostic path — FeatureGatingNetwork over stylometric features
      C. Fusion        — concat([text_emb, style_emb]) -> classifier
    """

    def __init__(self, config):
        super().__init__()

        model_cfg = config.get("model", {})
        data_cfg  = config.get("data", {})

        self.num_labels     = model_cfg.get("num_labels", 2)
        self.num_handcrafted = data_cfg.get("num_handcrafted_features", 11)
        base_model_name     = model_cfg.get("base_model", "microsoft/unixcoder-base")

        print(f"Hybrid Model | Backbone: {base_model_name} | Features: {self.num_handcrafted}")

        # A. Text backbone
        hf_config = AutoConfig.from_pretrained(base_model_name)
        hf_config.hidden_dropout_prob      = model_cfg.get("hidden_dropout", 0.2)
        hf_config.attention_probs_dropout_prob = model_cfg.get("hidden_dropout", 0.2)
        self.base_model = AutoModel.from_pretrained(base_model_name, config=hf_config)

        if model_cfg.get("gradient_checkpointing", False):
            self.base_model.gradient_checkpointing_enable()

        self.hidden_size = hf_config.hidden_size

        # B. Pathways
        self.pooler         = AttentionPooler(self.hidden_size)
        self.feature_encoder = FeatureGatingNetwork(
            input_dim=self.num_handcrafted, output_dim=128,
            dropout_rate=model_cfg.get("hidden_dropout", 0.2),
        )

        # C. Fusion + classifier
        fusion_dim = self.hidden_size + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, self.num_labels),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.feature_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, extra_features, labels=None):
        # A. Semantic path
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self.pooler(outputs.last_hidden_state, attention_mask)

        # B. Agnostic path
        style_emb = self.feature_encoder(extra_features)

        # C. Fusion + classification
        combined = torch.cat([text_emb, style_emb], dim=1)
        logits   = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits, labels.view(-1))

        return logits, loss, combined
