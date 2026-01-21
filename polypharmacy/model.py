from typing import List, Optional

import torch
from torch import nn


class PolypharmacyLSTMClassifier(nn.Module):
    def __init__(
        self,
        drug_embeddings: torch.Tensor,
        disease_embeddings: torch.Tensor,
        lstm_hidden_dim: int,
        mlp_hidden_dim: int,
        mlp_layers: int,
        dropout: float,
        freeze_kg: bool,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.drug_embedding = nn.Embedding.from_pretrained(
            drug_embeddings, freeze=freeze_kg, padding_idx=pad_idx
        )
        self.disease_embedding = nn.Embedding.from_pretrained(
            disease_embeddings, freeze=freeze_kg
        )
        self.lstm = nn.LSTM(
            input_size=drug_embeddings.size(1),
            hidden_size=lstm_hidden_dim,
            batch_first=True,
        )
        mlp_layers_seq: List[nn.Module] = []
        input_dim = lstm_hidden_dim + disease_embeddings.size(1)
        for _ in range(max(1, mlp_layers)):
            mlp_layers_seq.append(nn.Linear(input_dim, mlp_hidden_dim))
            mlp_layers_seq.append(nn.LayerNorm(mlp_hidden_dim))
            mlp_layers_seq.append(nn.ReLU())
            mlp_layers_seq.append(nn.Dropout(dropout))
            input_dim = mlp_hidden_dim
        mlp_layers_seq.append(nn.Linear(input_dim, 1))
        self.classifier = nn.Sequential(*mlp_layers_seq)

    def forward(
        self,
        drug_sequences: torch.Tensor,
        drug_lengths: torch.Tensor,
        disease_indices: torch.Tensor,
    ) -> torch.Tensor:
        drug_emb = self.drug_embedding(drug_sequences)
        packed = nn.utils.rnn.pack_padded_sequence(
            drug_emb,
            drug_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        # Final hidden state represents the drug-combination embedding.
        combo_embedding = h_n[-1]
        disease_emb = self.disease_embedding(disease_indices)
        # Concatenate drug-combo and disease embeddings before classification.
        combined = torch.cat([combo_embedding, disease_emb], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits
