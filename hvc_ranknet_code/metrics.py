import torch


def ndcg_at_k(
    rel,
    rank_pred,
    K,
):
    device = rel.device
    n = rel.numel()
    k = min(K, n)
    rel = rel.double()
    topk_indices = rank_pred[:k]
    gains = torch.exp2(rel[topk_indices]) - 1.0
    discounts = 1.0 / torch.log2(
        torch.arange(
            2,
            k + 2,
            device=device,
            dtype=torch.float64,
        )
    )
    dcg = (gains * discounts).sum()
    ideal_gains = torch.sort(rel, descending=True)[0][:k]
    idcg = (
        (torch.exp2(ideal_gains) - 1.0) * discounts
    ).sum().clamp_min(1e-6)
    return (dcg / idcg).item()
