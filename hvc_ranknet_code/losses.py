import math

import torch


def lambdarank_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    topk=1,
    eps: float = 1e-10,
) -> torch.Tensor:
    device = scores.device
    item_count = scores.numel()
    if item_count < 2:
        return scores.new_tensor(0.0)

    if topk is None or topk > item_count:
        topk = item_count

    stable_scores = scores.double()
    stable_labels = labels.double()
    ideal_order = torch.argsort(stable_labels, descending=True)
    ideal_gains = torch.exp2(stable_labels[ideal_order]) - 1.0
    discounts = torch.log2(
        torch.arange(
            2,
            topk + 2,
            device=device,
            dtype=torch.float64,
        )
    )
    ideal_dcg = (
        ideal_gains[:topk] / discounts
    ).sum().clamp_min(eps)

    gains = torch.exp2(stable_labels) - 1.0
    score_differences = (
        stable_scores.unsqueeze(1) - stable_scores.unsqueeze(0)
    )
    gain_differences = gains.unsqueeze(1) - gains.unsqueeze(0)
    pair_sign = torch.sign(gain_differences)
    ordered_pair_mask = pair_sign > 0
    if not ordered_pair_mask.any():
        return scores.new_tensor(0.0)

    predicted_order = torch.argsort(stable_scores, descending=True)
    inverse_rank = torch.empty_like(predicted_order)
    inverse_rank[predicted_order] = torch.arange(
        item_count,
        device=device,
        dtype=torch.long,
    )
    rank = inverse_rank.double() + 1.0
    in_topk = rank <= float(topk)
    topk_pair_mask = (
        in_topk.unsqueeze(1) | in_topk.unsqueeze(0)
    ) & ordered_pair_mask
    if not topk_pair_mask.any():
        return scores.new_tensor(0.0)

    reciprocal_discount = 1.0 / torch.log2(rank + 1.0)
    delta_dcg = gain_differences * (
        reciprocal_discount.unsqueeze(1)
        - reciprocal_discount.unsqueeze(0)
    )
    delta_ndcg = delta_dcg.abs() / ideal_dcg
    logistic_argument = -pair_sign * score_differences
    logistic_loss = torch.logaddexp(
        torch.zeros_like(logistic_argument),
        logistic_argument,
    ) / math.log(2.0)
    loss = (delta_ndcg * logistic_loss)[topk_pair_mask].sum()
    return loss.to(scores.dtype)
