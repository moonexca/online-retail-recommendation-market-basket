"""Offline evaluation of popularity and collaborative-filtering recommenders."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

SERVICE_CODES = {
    "POST",
    "BANK CHARGES",
    "D",
    "M",
    "DOT",
    "ADJUST2",
    "PADS",
    "C2",
    "SP1002",
    "TEST001",
}


def load_workbook(path: str | Path) -> pd.DataFrame:
    """Load every worksheet from the official UCI workbook."""

    sheets = pd.read_excel(path, sheet_name=None)
    return pd.concat(sheets.values(), ignore_index=True)


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the minimum rules needed for positive UK purchase behaviour."""

    data = frame.copy()
    data.columns = data.columns.str.strip().str.replace(" ", "_", regex=False)
    required = {
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer_ID",
        "Country",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["Invoice"] = data["Invoice"].astype(str).str.strip()
    data["StockCode"] = data["StockCode"].astype(str).str.strip()
    data["Customer_ID"] = pd.to_numeric(data["Customer_ID"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

    valid = (
        data["InvoiceDate"].notna()
        & data["Customer_ID"].notna()
        & data["Quantity"].gt(0)
        & data["Price"].gt(0)
        & data["Country"].eq("United Kingdom")
        & ~data["Invoice"].str.upper().str.startswith("C")
        & ~data["StockCode"].str.upper().isin(SERVICE_CODES)
    )
    columns = [
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Customer_ID",
    ]
    return data.loc[valid, columns].reset_index(drop=True)


def temporal_holdout(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each customer's last invoice; retain users with prior history."""

    invoices = (
        data.groupby(["Customer_ID", "Invoice"], as_index=False)["InvoiceDate"]
        .max()
        .sort_values(["Customer_ID", "InvoiceDate", "Invoice"])
    )
    invoice_counts = invoices.groupby("Customer_ID")["Invoice"].transform("count")
    eligible = invoices.loc[invoice_counts.ge(2), "Customer_ID"].unique()
    scoped = data[data["Customer_ID"].isin(eligible)].copy()
    last_invoice = (
        invoices[invoices["Customer_ID"].isin(eligible)]
        .groupby("Customer_ID", as_index=False)
        .tail(1)[["Customer_ID", "Invoice"]]
        .assign(is_test=True)
    )
    marked = scoped.merge(last_invoice, on=["Customer_ID", "Invoice"], how="left")
    marked["is_test"] = marked["is_test"].eq(True)
    test = marked[marked["is_test"]].drop(columns="is_test")
    train = marked[~marked["is_test"]].drop(columns="is_test")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _top_k(scores: np.ndarray, seen: np.ndarray, k: int) -> np.ndarray:
    adjusted = scores.copy()
    adjusted[seen] = -np.inf
    limit = min(k, np.isfinite(adjusted).sum())
    if limit == 0:
        return np.array([], dtype=int)
    candidate = np.argpartition(adjusted, -limit)[-limit:]
    return candidate[np.argsort(adjusted[candidate])[::-1]]


def _summarise(
    name: str,
    recommendations: list[np.ndarray],
    truths: list[set[int]],
    *,
    top_k: int,
    n_items: int,
) -> dict[str, float | int | str]:
    hits = [len(set(rec).intersection(truth)) for rec, truth in zip(recommendations, truths)]
    precision = [hit / top_k for hit in hits]
    recall = [hit / len(truth) for hit, truth in zip(hits, truths)]
    unique_recommended = set(np.concatenate(recommendations).tolist()) if recommendations else set()
    return {
        "model": name,
        "users_evaluated": len(truths),
        f"precision_at_{top_k}": float(np.mean(precision)),
        f"recall_at_{top_k}": float(np.mean(recall)),
        f"hit_rate_at_{top_k}": float(np.mean(np.asarray(hits) > 0)),
        f"catalog_coverage_at_{top_k}": len(unique_recommended) / n_items,
    }


def evaluate_recommenders(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    top_k: int = 10,
    n_components: int = 64,
    min_item_interactions: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare global popularity with truncated-SVD collaborative filtering."""

    item_frequency = train.groupby("StockCode")["Customer_ID"].nunique()
    valid_items = item_frequency[item_frequency.ge(min_item_interactions)].index
    train = train[train["StockCode"].isin(valid_items)].copy()
    test = test[test["StockCode"].isin(valid_items)].copy()

    users = sorted(set(train["Customer_ID"]).intersection(test["Customer_ID"]))
    items = sorted(train["StockCode"].unique())
    user_to_index = {user: index for index, user in enumerate(users)}
    item_to_index = {item: index for index, item in enumerate(items)}

    interactions = (
        train[train["Customer_ID"].isin(users)]
        .groupby(["Customer_ID", "StockCode"], as_index=False)["Quantity"]
        .sum()
    )
    rows = interactions["Customer_ID"].map(user_to_index).to_numpy()
    cols = interactions["StockCode"].map(item_to_index).to_numpy()
    values = np.log1p(interactions["Quantity"].to_numpy(dtype=float))
    matrix = csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))

    test_items = (
        test[test["Customer_ID"].isin(users)]
        .groupby("Customer_ID")["StockCode"]
        .agg(lambda values: set(values))
    )
    evaluation_users: list[int] = []
    truths: list[set[int]] = []
    for user in users:
        user_index = user_to_index[user]
        seen = set(matrix[user_index].indices.tolist())
        truth = {
            item_to_index[item]
            for item in test_items.get(user, set())
            if item in item_to_index and item_to_index[item] not in seen
        }
        if truth:
            evaluation_users.append(user_index)
            truths.append(truth)
    if not truths:
        raise ValueError("No users have novel holdout items after filtering")

    popularity_scores = np.asarray(matrix.getnnz(axis=0), dtype=float)
    popularity_recommendations = [
        _top_k(popularity_scores, matrix[index].indices, top_k)
        for index in evaluation_users
    ]

    components = max(1, min(n_components, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=components, random_state=random_state)
    user_factors = svd.fit_transform(matrix)
    svd_recommendations = []
    for index in evaluation_users:
        scores = user_factors[index] @ svd.components_
        svd_recommendations.append(_top_k(scores, matrix[index].indices, top_k))

    rows_out = [
        _summarise(
            "Popularity baseline",
            popularity_recommendations,
            truths,
            top_k=top_k,
            n_items=len(items),
        ),
        _summarise(
            "Truncated SVD",
            svd_recommendations,
            truths,
            top_k=top_k,
            n_items=len(items),
        ),
    ]
    return pd.DataFrame(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/online_retail_II.xlsx")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--min-item-interactions", type=int, default=10)
    args = parser.parse_args()

    raw = load_workbook(args.data)
    cleaned = clean_transactions(raw)
    train, test = temporal_holdout(cleaned)
    metrics = evaluate_recommenders(
        train,
        test,
        top_k=args.top_k,
        n_components=args.components,
        min_item_interactions=args.min_item_interactions,
    )
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics.round(5).to_csv(destination / "evaluation_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "clean_transactions": len(cleaned),
                "train_transactions": len(train),
                "holdout_transactions": len(test),
                "holdout_rule": "last invoice per customer",
            }
        ]
    ).to_csv(destination / "evaluation_cohort.csv", index=False)
    print(metrics.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
