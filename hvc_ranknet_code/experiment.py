import csv
import os
import random

import numpy as np
import torch

from .train_one_run import train_one_run


def train_multiple_runs_ranknet(
    model_fn,
    train_loader_fn,
    test_loader_fn,
    device="cpu",
    epochs=50,
    n_runs=30,
    lr=1e-4,
    csv_raw="raw_epoch_data.csv",
    csv_stat="stats_epoch_mean_std.csv",
    seeds=None,
    topk=1,
    test_every=1,
    problem="WFG",
    num_obj=5,
):
    if seeds is None:
        seeds = [index * 1234 + 7 for index in range(n_runs)]

    for path in (csv_raw, csv_stat):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(
        "..",
        "Models",
        f"rank_{problem}_{num_obj}.pth",
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(csv_raw, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "run_id",
                "epoch",
                "train_loss",
                "test_loss",
                "Pk",
                "NDCG1",
            ]
        )

    all_train_loss = []
    all_test_loss = []
    all_precision_at_k = []
    all_ndcg_at_k = []

    for run_index, seed in enumerate(seeds, 1):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loader = train_loader_fn()
        test_loader = test_loader_fn()
        history = train_one_run(
            model,
            train_loader,
            test_loader,
            optimizer,
            device,
            epochs,
            topk=topk,
            test_every=test_every,
            log_prefix=f"[Run {run_index}/{n_runs}]",
        )

        with open(csv_raw, "a", newline="") as output:
            writer = csv.writer(output)
            for epoch in range(epochs):
                writer.writerow(
                    [
                        run_index,
                        epoch + 1,
                        history["train_loss"][epoch],
                        history["test_loss"][epoch],
                        history["pk"][epoch],
                        history["ndcg"][epoch],
                    ]
                )

        all_train_loss.append(history["train_loss"])
        all_test_loss.append(history["test_loss"])
        all_precision_at_k.append(history["pk"])
        all_ndcg_at_k.append(history["ndcg"])
        torch.save(model.state_dict(), model_path)

    train_loss_array = np.array(all_train_loss)
    test_loss_array = np.array(all_test_loss)
    precision_at_k_array = np.array(all_precision_at_k)
    ndcg_at_k_array = np.array(all_ndcg_at_k)

    with open(csv_stat, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "epoch",
                "train_mean",
                "train_lower",
                "train_upper",
                "",
                "test_mean",
                "test_lower",
                "test_upper",
                "",
                "p1_mean",
                "p1_lower",
                "p1_upper",
                "",
                "ndcg_mean",
                "ndcg_lower",
                "ndcg_upper",
            ]
        )

        for epoch in range(epochs):
            def row(values):
                mean = np.nanmean(values[:, epoch])
                std = np.nanstd(values[:, epoch])
                return mean, mean - std, mean + std

            train_bounds = row(train_loss_array)
            test_bounds = row(test_loss_array)
            precision_bounds = row(precision_at_k_array)
            ndcg_bounds = row(ndcg_at_k_array)
            writer.writerow(
                [
                    epoch + 1,
                    *train_bounds,
                    "",
                    *test_bounds,
                    "",
                    *precision_bounds,
                    "",
                    *ndcg_bounds,
                ]
            )

    print(f"√ Raw data → {csv_raw}\n√ Mean/Std  → {csv_stat}")
