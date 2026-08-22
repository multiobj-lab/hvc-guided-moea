import torch
import torch.nn as nn


class HVCRankNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.cand_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.sel_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidates: torch.Tensor,
        cand_mask: torch.Tensor,
        selected: torch.Tensor,
        sel_mask: torch.Tensor,
    ):
        candidate_embeddings = self.cand_enc(candidates)
        selected_embeddings = self.sel_enc(selected)
        context = (
            selected_embeddings * sel_mask.unsqueeze(-1)
        ).sum(dim=1)
        combined = torch.cat(
            (
                candidate_embeddings,
                context.unsqueeze(1).expand_as(candidate_embeddings),
            ),
            dim=-1,
        )
        scores = self.head(combined).squeeze(-1)
        scores = scores.masked_fill(~cand_mask, -1e9)
        return scores
