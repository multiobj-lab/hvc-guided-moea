from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class QueryDataset(Dataset):
    def __init__(
        self,
        candidates: List[np.ndarray],
        selected_sets: List[np.ndarray],
        labels: List[np.ndarray],
    ):
        assert len(candidates) == len(selected_sets) == len(labels)
        self.candidates = [
            torch.as_tensor(candidate, dtype=torch.float32)
            for candidate in candidates
        ]
        self.selected_sets = [
            torch.as_tensor(selected, dtype=torch.float32)
            for selected in selected_sets
        ]
        self.labels = [
            torch.as_tensor(label, dtype=torch.float32)
            for label in labels
        ]

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "candidates": self.candidates[idx],
            "selected": self.selected_sets[idx],
            "labels": self.labels[idx],
        }


def rank_with_ties(
    values: Union[Sequence[float], np.ndarray],
    descending: bool = True,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).flatten()
    candidate_count = values.size
    unique_values = np.unique(values)
    unique_values = (
        np.sort(unique_values)[::-1]
        if descending
        else np.sort(unique_values)
    )
    value_to_rank = {
        value: index + 1 for index, value in enumerate(unique_values)
    }
    ranks = np.array(
        [value_to_rank[value] for value in values],
        dtype=np.int32,
    )
    relevance = (candidate_count + 1 - ranks) ** 1
    return relevance


def collate_double_pad(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    vector_dim = batch[0]["candidates"].shape[1]
    max_candidates = max(item["candidates"].shape[0] for item in batch)
    max_selected = max(item["selected"].shape[0] for item in batch)

    candidate_pad = batch[0]["candidates"].new_zeros(
        batch_size,
        max_candidates,
        vector_dim,
    )
    candidate_mask = torch.zeros(
        batch_size,
        max_candidates,
        dtype=torch.bool,
    )
    label_pad = batch[0]["labels"].new_zeros(
        batch_size,
        max_candidates,
    )

    selected_pad = batch[0]["selected"].new_zeros(
        batch_size,
        max_selected,
        vector_dim,
    )
    selected_mask = torch.zeros(
        batch_size,
        max_selected,
        dtype=torch.bool,
    )

    for index, item in enumerate(batch):
        candidate_count = item["candidates"].shape[0]
        selected_count = item["selected"].shape[0]
        candidate_pad[index, :candidate_count] = item["candidates"]
        candidate_mask[index, :candidate_count] = True
        label_pad[index, :candidate_count] = item["labels"]
        selected_pad[index, :selected_count] = item["selected"]
        selected_mask[index, :selected_count] = True

    return {
        "candidates": candidate_pad,
        "cand_mask": candidate_mask,
        "labels": label_pad,
        "selected": selected_pad,
        "sel_mask": selected_mask,
    }
