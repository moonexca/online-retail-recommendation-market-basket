# Recommendation Systems and Market Basket Analysis in Online Retail

Academic machine learning and data-visualisation project by **Soledad Yash** analysing online retail transactions through recommendation systems, association rule mining, and dashboard-oriented business insights.

## Overview

This repository contains a historical academic project based on the **Online Retail II** transactional dataset. The work explores how customer-product interactions can be transformed into practical analytical outputs for retail decision making.

The project combines:

- transaction cleaning and restructuring
- recommendation-system logic
- collaborative filtering concepts
- market basket analysis
- **Apriori** and **FP-Growth**
- dashboard-oriented business interpretation

The overall goal is to show how retail transaction data can support product recommendation, cross-selling, merchandising, and customer-behaviour insight.

## Project Goals

The analysis focuses on:

- understanding repeated purchase behaviour in transactional retail data
- building user-item logic for recommendation use cases
- identifying frequent itemsets and association rules
- comparing Apriori and FP-Growth in an applied setting
- translating results into business-facing visual and strategic insights

## Methods

The project includes:

- cleaning of retail transactions
- exclusion of refunds, invalid rows, and non-product noise
- transformation into recommendation-ready interaction structures
- collaborative filtering concepts using item and user similarity
- association rule mining using:
  - **Apriori**
  - **FP-Growth**
- visualisation and dashboard-oriented summarisation

## Repository Contents

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

## Data and Storage Notes

This portfolio version intentionally excludes several very large intermediate or binary artifacts from public publication, including:

- full raw retail data files
- similarity matrices in `.pkl` format
- compressed archive duplicates
- notebook checkpoints
- redundant notebook variants

This keeps the repository practical for GitHub while preserving the core academic deliverables.

See [DATASET_AND_STORAGE_NOTES.md](./DATASET_AND_STORAGE_NOTES.md) for more detail.

## Reproducibility Notes

- This repository is published as a **historical academic project**.
- It has been cleaned for portfolio use and reduced to the core files most relevant for review.
- The notebook reflects an assignment-era workflow rather than a production-grade package.
- Some outputs and modelling choices should be interpreted in the context of coursework constraints and the original dataset size.

## Academic and IP Statement

This repository contains academic work authored by **Soledad Yash** and is published for portfolio and research-communication purposes.

- The modelling, interpretation, and project structure are presented as the author's academic work.
- The underlying dataset and any external sources remain subject to their original ownership and usage conditions.
- Mention of the academic institution provides context only and does not imply endorsement.

See [ACADEMIC_USE_AND_IP.md](./ACADEMIC_USE_AND_IP.md) for the extended statement.

## AI Use Disclosure

AI-assisted support may have been used in limited surrounding tasks such as wording refinement, structural editing support, or explanation assistance. Final responsibility for validation, interpretation, and publication remains with **Soledad Yash**.

See [AI_USE_DISCLOSURE.md](./AI_USE_DISCLOSURE.md) for a fuller note on limitations, responsible use, and security awareness.

## Portfolio Relevance

This project supports a portfolio narrative around:

- data analytics
- recommendation systems
- customer behaviour analysis
- retail machine learning
- dashboard-oriented business communication

## Author

**Soledad Yash**  
Dublin, Ireland  
[LinkedIn](https://www.linkedin.com/in/soledad-yash)  
[GitHub](https://github.com/asolyash)
