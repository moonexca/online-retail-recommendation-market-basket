# Online Retail Recommendation and Market Basket Analysis

A reproducible analysis of UK retail transactions, combining association-rule
mining with offline evaluation of recommendation strategies.

## At a glance

- **Problem:** recommend products from implicit purchase behaviour and identify
  recurring basket relationships.
- **Data:** UCI Online Retail II, covering transactions from 2009–2011.
- **Methods:** transaction cleaning, last-invoice temporal holdout, popularity
  baseline, truncated-SVD collaborative filtering, Apriori, and FP-Growth.
- **Evaluation:** Precision@10, Recall@10, Hit Rate@10, and catalogue coverage.
- **Stack:** Python, pandas, scipy, scikit-learn, mlxtend, Plotly.

## Headline result

The original academic notebook demonstrated recommendation logic but did not
measure out-of-sample quality. The portfolio pipeline now holds out each eligible
customer’s final invoice and compares collaborative filtering against a simple
popularity baseline. Exact results are committed in
[`results/evaluation_metrics.csv`](results/evaluation_metrics.csv).

This matters because a recommender is not demonstrated merely by producing a
list: it should beat a transparent baseline on unseen behaviour.

On 3,400 evaluable customers, truncated SVD improved Precision@10 from `0.02956`
to `0.04906`, Recall@10 from `0.02625` to `0.05532`, and Hit Rate@10 from
`0.23118` to `0.32235`. Catalogue coverage increased from `2.03%` to `22.75%`.
The gains are meaningful, while the low absolute precision also keeps the result
appropriately modest.

## Reproduce the evaluation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python -m src.evaluation
pytest -q
```

The download script retrieves the workbook from the official UCI repository.
Use `python scripts/download_data.py --write-csv` to generate the CSV filename
expected by the historical notebook.

## Repository map

- [`src/evaluation.py`](src/evaluation.py): cleaning, temporal split, recommenders,
  and ranking metrics
- [`scripts/download_data.py`](scripts/download_data.py): reproducible source-data
  retrieval
- [`tests/test_evaluation.py`](tests/test_evaluation.py): end-to-end toy-data test
- [`results/`](results): full-data evaluation metrics and cohort description
- [`ML_DV.ipynb`](ML_DV.ipynb): historical academic analysis and visualisations
- `frequent_itemsets_*.csv`: small derived outputs from association-rule mining
- [`DATASET_AND_STORAGE_NOTES.md`](DATASET_AND_STORAGE_NOTES.md): storage rationale

## Evaluation choices and limitations

- A customer’s final invoice is held out, preventing future purchases from
  informing their training history.
- Previously purchased items are excluded from the held-out target and ranking.
- Offline ranking metrics do not measure revenue lift, diversity, fairness, or
  causal business impact.
- The data is historical and UK-heavy; results should not be treated as current
  retail behaviour.

## Author

**Soledad Yash** · Dublin, Ireland<br>
[LinkedIn](https://www.linkedin.com/in/soledad-yash) ·
[GitHub](https://github.com/moonexca)

Academic context and responsible-use notes are available in
[`ACADEMIC_USE_AND_IP.md`](ACADEMIC_USE_AND_IP.md) and
[`AI_USE_DISCLOSURE.md`](AI_USE_DISCLOSURE.md).
