# Recommendation Systems and Market Basket Analysis in Online Retail

Retail data has a way of appearing ordinary until one looks long enough. Then repetition begins to surface. Some products travel together. Some customers form habits that resemble signatures. Certain items do not merely sell; they attract one another. What looks like noise at the level of transaction becomes pattern at the level of sequence. This project was built to read that pattern.

This repository contains an academic machine learning and data-visualisation project by **Soledad Yash** based on the **Online Retail II** dataset. It explores how transactional retail data can support recommendation logic, association rule mining, and practical business insight.

## What this project does

The work moves through the retail dataset in stages:

- it cleans transactions and removes invalid or misleading records
- it restructures purchase information into recommendation-ready forms
- it studies customer-product interaction through collaborative filtering logic
- it applies **Apriori** and **FP-Growth** to identify repeated product associations
- it translates those outputs into business-facing interpretation

The project is less interested in technical display for its own sake than in the small intelligence hidden in repeated behaviour.

## Main analytical components

- transaction cleaning and restructuring
- recommendation-system logic
- collaborative filtering concepts
- market basket analysis
- **Apriori**
- **FP-Growth**
- dashboard-oriented business interpretation

## Repository contents

- [ML_DV.ipynb](./ML_DV.ipynb): main notebook selected for portfolio presentation
- [Report_ML_DVZ.odt](./Report_ML_DVZ.odt): written report
- [frequent_itemsets_apriori.csv](./frequent_itemsets_apriori.csv): small derived output
- [frequent_itemsets_fpgrowth.csv](./frequent_itemsets_fpgrowth.csv): small derived output

## Environment

The notebook uses Python packages including:

- `pandas`
- `numpy`
- `matplotlib`
- `plotly`
- `scikit-learn`
- `mlxtend`
- `dash`
- `jupyter-dash`

See [requirements.txt](./requirements.txt) for a compact environment list.

## Data and storage notes

This portfolio version intentionally excludes several very large intermediate or binary artifacts from public publication, including:

- full raw retail data files
- similarity matrices stored as `.pkl`
- compressed duplicate archives
- notebook checkpoints
- redundant notebook variants

This keeps the repository readable, lighter, and more realistic for public review.

See [DATASET_AND_STORAGE_NOTES.md](./DATASET_AND_STORAGE_NOTES.md) for more detail.

## Why it matters

This project reflects a set of interests that recur in my work:

- customer behaviour seen through structure rather than anecdote
- machine learning used in commercial settings without losing interpretability
- pattern detection as a basis for better decisions
- analytics that can move from data preparation to business narrative

## Reproducibility notes

- This repository is published as a **historical academic project**.
- It has been cleaned for portfolio use and reduced to the core files most relevant for review.
- The notebook reflects an assignment-era workflow rather than a production package.

## Academic and IP statement

This repository contains academic work authored by **Soledad Yash** and is published for portfolio and research communication.

- The modelling, interpretation, and project structure are presented as the author's academic work.
- The underlying dataset and any external sources remain subject to their original ownership and usage conditions.
- The academic context is acknowledged for transparency only and does not imply endorsement.

See [ACADEMIC_USE_AND_IP.md](./ACADEMIC_USE_AND_IP.md) for the extended statement.

## AI use disclosure

AI-assisted support may have been used in limited surrounding tasks such as wording refinement, structural editing support, or explanation assistance. Final responsibility for validation, interpretation, and publication remains with **Soledad Yash**.

See [AI_USE_DISCLOSURE.md](./AI_USE_DISCLOSURE.md) for a fuller note on limitations, responsible use, and security awareness.

## Author

**Soledad Yash**  
Dublin, Ireland  
[LinkedIn](https://www.linkedin.com/in/soledad-yash)  
[GitHub](https://github.com/moonexca)
