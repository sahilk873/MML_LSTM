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
        disease_token_position: Optional[str] = None,
        concat_disease_after_lstm: bool = True,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        valid_positions = {None, "first", "last"}
        if disease_token_position not in valid_positions:
            raise ValueError(
                "disease_token_position must be one of None, 'first', or 'last'."
            )
        self.drug_embedding = nn.Embedding.from_pretrained(
            drug_embeddings, freeze=freeze_kg, padding_idx=pad_idx
        )
        self.disease_embedding = nn.Embedding.from_pretrained(
            disease_embeddings, freeze=freeze_kg
        )
        self.disease_token_position = disease_token_position
        self.concat_disease_after_lstm = concat_disease_after_lstm
        self.disease_to_drug = None
        if disease_token_position is not None and disease_embeddings.size(1) != drug_embeddings.size(1):
            self.disease_to_drug = nn.Linear(
                disease_embeddings.size(1), drug_embeddings.size(1)
            )
        self.lstm = nn.LSTM(
            input_size=drug_embeddings.size(1),
            hidden_size=lstm_hidden_dim,
            batch_first=True,
        )
        mlp_layers_seq: List[nn.Module] = []
        input_dim = lstm_hidden_dim
        if self.concat_disease_after_lstm:
            input_dim += disease_embeddings.size(1)
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
        disease_emb = self.disease_embedding(disease_indices)
        if self.disease_token_position is not None:
            disease_token = disease_emb
            if self.disease_to_drug is not None:
                disease_token = self.disease_to_drug(disease_token)
            disease_token = disease_token.unsqueeze(1)
            if self.disease_token_position == "first":
                drug_emb = torch.cat([disease_token, drug_emb], dim=1)
            else:
                drug_emb = torch.cat([drug_emb, disease_token], dim=1)
            drug_lengths = drug_lengths + 1
        packed = nn.utils.rnn.pack_padded_sequence(
            drug_emb,
            drug_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        # Final hidden state represents the drug-combination embedding.
        combo_embedding = h_n[-1]
        # Concatenate drug-combo and disease embeddings before classification.
        if self.concat_disease_after_lstm:
            combined = torch.cat([combo_embedding, disease_emb], dim=1)
        else:
            combined = combo_embedding
        logits = self.classifier(combined).squeeze(1)
        return logits
