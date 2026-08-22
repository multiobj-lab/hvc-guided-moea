from datetime import datetime

import numpy as np

from .training import run_epoch


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def train_one_run(
    model,
    train_loader,
    test_loader,
    optimizer,
    device,
    epochs,
    topk=1,
    test_every=1,
    log_prefix="",
):
    history = {
        "train_loss": [],
        "test_loss": [],
        "pk": [],
        "ndcg": [],
    }

    if test_every < 1:
        raise ValueError(
            f"test_every must be at least 1, got {test_every}"
        )

    for epoch in range(1, epochs + 1):
        (
            train_loss,
            train_precision_at_1,
            _,
            _,
            train_ndcg_at_k,
            train_precision_at_k_std,
            train_ndcg_at_k_std,
        ) = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            topk=topk,
        )
        history["train_loss"].append(train_loss)
        do_test = epoch % test_every == 0

        if do_test:
            (
                test_loss,
                test_precision_at_1,
                test_precision_at_k,
                _,
                test_ndcg_at_k,
                test_precision_at_k_std,
                test_ndcg_at_k_std,
            ) = run_epoch(
                model,
                test_loader,
                None,
                device,
                topk=topk,
            )
            history["test_loss"].append(test_loss)
            history["pk"].append(test_precision_at_k)
            history["ndcg"].append(test_ndcg_at_k)
            print(
                f"[{get_timestamp()}] {log_prefix} E{epoch:03d}: "
                f"Ltr={train_loss:.4f}  Lte={test_loss:.4f}  "
                f"Pk={test_precision_at_k:.3f}±"
                f"{test_precision_at_k_std:.3f}  "
                f"N@k={test_ndcg_at_k:.3f}±"
                f"{test_ndcg_at_k_std:.3f}"
            )
        else:
            history["test_loss"].append(np.nan)
            history["pk"].append(np.nan)
            history["ndcg"].append(np.nan)
            print(
                f"[{get_timestamp()}] {log_prefix} E{epoch:03d}: "
                f"Ltr={train_loss:.4f}  [Skip Test]"
            )

    return history
