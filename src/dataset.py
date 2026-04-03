"""
AgnosticDataset — hybrid dataset for SemEval Task A.
Handles tokenization, feature loading, normalization, and random cropping.
Used by train.py.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class AgnosticDataset(Dataset):
    """
    Hybrid dataset: semantic tokens from UnixCoder + normalized stylometric features.

    Normalisation applied to extra_features at __getitem__ time:
      indices 0, 1, 7  -> log1p
      all indices       -> clamp [0, 100]

    Random cropping (train only): random start offset when total_len > max_length.
    """

    def __init__(self, dataframe, tokenizer, max_length: int = 512, is_train: bool = False):
        from src.config import DEFAULT_CONFIG

        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.is_train    = is_train

        self.df = dataframe.reset_index(drop=True)

        required = {'code', 'label', 'agnostic_features'}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        features_list    = self.df['agnostic_features'].tolist()
        self.features_matrix = np.array(features_list, dtype=np.float32)

        self.num_samples  = len(self.df)
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

        print(f"Dataset | Split: {'TRAIN' if is_train else 'VAL/TEST'} | Samples: {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def _normalize_features(self, feature_vector: torch.Tensor) -> torch.Tensor:
        """
        Normalise feature vector before feeding to FeatureGatingNetwork.

        idx 0 (perplexity)  -> log1p
        idx 1 (id_len_avg)  -> log1p
        idx 7 (line_len_std)-> log1p
        all                  -> clamp [0, 100]
        """
        x = feature_vector.clone()
        for i in [0, 1, 7]:
            if i < x.shape[0]:
                x[i] = torch.log1p(x[i])
        return torch.clamp(x, min=0.0, max=100.0)

    def __getitem__(self, idx):
        code  = str(self.df.iat[idx, self.df.columns.get_loc('code')])
        label = int(self.df.iat[idx, self.df.columns.get_loc('label')])

        raw_feats    = self.features_matrix[idx]
        feats_tensor = torch.tensor(raw_feats, dtype=torch.float32)
        norm_feats   = self._normalize_features(feats_tensor)

        # Tokenise
        input_ids = self.tokenizer.encode(code, add_special_tokens=True, truncation=False)
        total_len = len(input_ids)

        # Random crop (train) or head-truncate (eval) when over max_length
        if total_len > self.max_length:
            if self.is_train:
                start_token  = input_ids[0]
                max_start    = total_len - self.max_length + 1
                rand_start   = int(np.random.randint(1, max_start))
                final_ids    = [start_token] + input_ids[rand_start : rand_start + self.max_length - 1]
            else:
                final_ids = input_ids[:self.max_length]
        else:
            final_ids = input_ids

        processed_len  = len(final_ids)
        padding_needed = self.max_length - processed_len

        if padding_needed > 0:
            final_ids      += [self.pad_token_id] * padding_needed
            attention_mask  = [1] * processed_len + [0] * padding_needed
        else:
            attention_mask  = [1] * self.max_length

        return {
            "input_ids":      torch.tensor(final_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "extra_features": norm_feats,
            "labels":         torch.tensor(label, dtype=torch.long),
        }
