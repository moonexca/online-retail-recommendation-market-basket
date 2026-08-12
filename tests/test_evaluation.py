import pandas as pd

from src.evaluation import clean_transactions, evaluate_recommenders, temporal_holdout


def _toy_transactions() -> pd.DataFrame:
    rows = []
    patterns = {
        1: (("A", "C"), "B"),
        2: (("B", "C"), "A"),
        3: (("A", "B"), "C"),
        4: (("A", "D"), "B"),
        5: (("B", "D"), "A"),
        6: (("B", "C"), "D"),
    }
    for customer, (history, heldout) in patterns.items():
        for position, item in enumerate(history):
            rows.append(
                {
                    "Invoice": f"{customer}-train-{position}",
                    "StockCode": item,
                    "Quantity": 1,
                    "InvoiceDate": f"2020-01-0{position + 1}",
                    "Price": 1.0,
                    "Customer ID": customer,
                    "Country": "United Kingdom",
                }
            )
        rows.append(
            {
                "Invoice": f"{customer}-test",
                "StockCode": heldout,
                "Quantity": 1,
                "InvoiceDate": "2020-02-01",
                "Price": 1.0,
                "Customer ID": customer,
                "Country": "United Kingdom",
            }
        )
    return pd.DataFrame(rows)


def test_temporal_evaluation_returns_baseline_and_svd():
    clean = clean_transactions(_toy_transactions())
    train, test = temporal_holdout(clean)
    metrics = evaluate_recommenders(
        train,
        test,
        top_k=2,
        n_components=2,
        min_item_interactions=1,
    )

    assert set(metrics["model"]) == {"Popularity baseline", "Truncated SVD"}
    assert metrics["users_evaluated"].min() > 0
    for column in ["precision_at_2", "recall_at_2", "hit_rate_at_2"]:
        assert metrics[column].between(0, 1).all()
