import math

import torch

from .losses import lambdarank_loss
from .metrics import ndcg_at_k

def run_epoch(
    model,
    loader,
    optimizer,
    device,
    topk=1,
):
    model.train(optimizer is not None)
    total_loss = 0.0
    query_count = 0
    precision_at_1_sum = 0
    precision_at_k_sum = 0
    exact_hit_at_k_sum = 0
    ndcg_at_k_sum = 0
    precision_at_k_squared_sum = 0.0
    ndcg_at_k_squared_sum = 0.0

    for batch in loader:
        candidates = batch["candidates"].to(device)
        candidate_mask = batch["cand_mask"].to(device)
        selected = batch["selected"].to(device)
        selected_mask = batch["sel_mask"].to(device)
        ranks = batch["labels"].to(device)
        scores = model(
            candidates,
            candidate_mask,
            selected,
            selected_mask,
        )
        batch_size, _ = scores.shape
        batch_loss = 0.0
        effective_queries = 0
        batch_precision_at_1 = 0
        batch_precision_at_k = 0
        batch_exact_hit_at_k = 0
        batch_ndcg_at_k = 0

        for index in range(batch_size):
            valid = candidate_mask[index]
            if valid.sum() < 2:
                continue

            query_scores = scores[index][valid]
            query_relevance = ranks[index][valid]
            candidate_count = query_relevance.size(0)
            if (
                query_relevance.max().item()
                == query_relevance.min().item()
                and optimizer is not None
            ):
                continue

            batch_loss += lambdarank_loss(
                query_scores,
                query_relevance,
                topk=topk,
            )
            effective_queries += 1
            predicted_order = torch.argsort(
                query_scores,
                descending=True,
            )
            batch_precision_at_1 += 1
            predicted_set = set(
                predicted_order[:topk].cpu().tolist()
            )
            k = min(topk, candidate_count)
            topk_values, topk_indices = torch.topk(
                query_relevance,
                k,
            )
            threshold = topk_values[-1]
            true_set = set(
                (query_relevance >= threshold)
                .nonzero(as_tuple=False)
                .flatten()
                .cpu()
                .tolist()
            )
            precision_at_k = len(predicted_set & true_set) / topk
            batch_precision_at_k += precision_at_k
            precision_at_k_squared_sum += precision_at_k ** 2
            batch_exact_hit_at_k += (
                1.0 if predicted_set == true_set else 0.0
            )
            ndcg = ndcg_at_k(
                query_relevance,
                predicted_order,
                K=topk,
            )
            batch_ndcg_at_k += ndcg
            ndcg_at_k_squared_sum += ndcg ** 2

        if effective_queries:
            if optimizer is not None:
                optimizer.zero_grad()
                (batch_loss / effective_queries).backward()
                optimizer.step()
            total_loss += batch_loss.item()
            query_count += effective_queries
            precision_at_1_sum += batch_precision_at_1
            precision_at_k_sum += batch_precision_at_k
            exact_hit_at_k_sum += batch_exact_hit_at_k
            ndcg_at_k_sum += batch_ndcg_at_k

    mean_loss = total_loss / query_count
    precision_at_1 = precision_at_1_sum / max(query_count, 1)
    precision_at_k = precision_at_k_sum / max(query_count, 1)
    exact_hit_at_k = exact_hit_at_k_sum / max(query_count, 1)
    mean_ndcg_at_k = ndcg_at_k_sum / max(query_count, 1)
    precision_at_k_std = math.sqrt(
        max(
            precision_at_k_squared_sum / query_count
            - precision_at_k ** 2,
            0.0,
        )
    )
    ndcg_at_k_std = math.sqrt(
        max(
            ndcg_at_k_squared_sum / query_count
            - mean_ndcg_at_k ** 2,
            0.0,
        )
    )
    return (
        mean_loss,
        precision_at_1,
        precision_at_k,
        exact_hit_at_k,
        mean_ndcg_at_k,
        precision_at_k_std,
        ndcg_at_k_std,
    )
