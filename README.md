# Delivery Delay Prediction — Olist E-Commerce

**MVP — Machine Learning & Analytics**
**PUC-Rio · Postgraduate Program in Machine Learning and Data Science**

| | |
|---|---|
| **Author** | Denilson Marcos — `4052025001353` |
| **Dataset** | Brazilian E-Commerce Public Dataset by Olist (Kaggle / GitHub mirror) |
| **Task** | Binary Classification — predict whether an order will be delivered **late** |
| **Environment** | Google Colab (Python 3.10) |

> **Reproducibility note:** All data is loaded directly from public GitHub raw URLs. No API keys, tokens, manual uploads, or local configuration are required. Run all cells from top to bottom: `Runtime → Run all`.

---

## 1. Overview

This project builds an end-to-end machine learning pipeline to predict **delivery delays** in the Olist Brazilian e-commerce marketplace. Given the attributes available at (or shortly after) purchase time, the model estimates whether an order will arrive **after** its estimated delivery date.

Accurate delay prediction has direct business value: it enables proactive customer communication, smarter logistics allocation, and early intervention on at-risk orders — all of which protect customer satisfaction and reduce support costs.

The notebook is fully self-contained and reproducible: clone-free, upload-free, and runnable end-to-end in a fresh Google Colab session.

---

## 2. Business Problem

Olist customers receive an **estimated delivery date** at checkout. When the actual delivery exceeds that estimate, customer satisfaction (and review scores) drop sharply.

**Goal:** train a binary classifier that flags orders likely to be delivered late, so the business can act *before* the delay materializes.

**Target variable**

```
is_late = 1  if  order_delivered_customer_date > order_estimated_delivery_date
is_late = 0  otherwise
```

---

## 3. Dataset

The **Brazilian E-Commerce Public Dataset by Olist** contains ~100k orders placed between 2016 and 2018, distributed across multiple relational tables (orders, items, payments, reviews, customers, sellers, products, geolocation).

| Table | Role in this project |
|---|---|
| `olist_orders_dataset` | Order lifecycle timestamps + delivery dates (core of the target) |
| `olist_order_items_dataset` | Items, price, freight value |
| `olist_products_dataset` | Product dimensions, weight, category |
| `olist_customers_dataset` | Customer location (state / region) |
| `olist_sellers_dataset` | Seller location (state / region) |

All tables are loaded directly from public **GitHub raw URLs** — no Kaggle credentials needed.

---

## 4. Methodology

1. **Data ingestion** — load all required tables from public raw URLs.
2. **Joining & assembly** — merge orders, items, products, customers, and sellers into a single analytical table.
3. **Target engineering** — derive `is_late` from delivery vs. estimated dates; drop orders without a valid delivery timestamp (canceled / undelivered).
4. **Feature engineering** — examples:
   - freight value, price, freight-to-price ratio
   - product weight and volume (`length × height × width`)
   - number of items per order
   - estimated delivery window (estimated date − purchase date)
   - customer/seller state and whether they share the same state/region
   - temporal features (purchase month, day of week, hour)
5. **Preprocessing** — handle missing values, encode categoricals, scale numeric features where required.
6. **Train/test split** — stratified split to preserve the class balance of `is_late`.
7. **Modeling** — train and compare baseline and tree-based classifiers (e.g. Logistic Regression, Random Forest, Gradient Boosting).
8. **Evaluation** — assess with metrics suited to imbalanced classification.

---

## 5. Evaluation Metrics

Because late deliveries are the **minority class**, accuracy alone is misleading. The notebook reports:

- **Precision / Recall / F1** (with emphasis on the positive `is_late` class)
- **ROC-AUC**
- **Confusion matrix**
- **Classification report**

> Recall on the *late* class is prioritized: missing a genuinely late order is more costly to the business than a false alarm.

---

## 6. How to Run

### Option A — Google Colab (recommended)

1. Open the notebook in Google Colab.
2. Click **`Runtime → Run all`**.
3. Wait for all cells to execute top to bottom.

No setup, no uploads, no credentials.

### Option B — Local / Jupyter

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

**Core dependencies**

```
python >= 3.10
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 7. Project Structure

```
.
├── delivery_delay_prediction.ipynb   # Main notebook (runs top to bottom)
├── README.md                         # This file
└── requirements.txt                  # Optional, for local execution
```

> All data is fetched at runtime from public GitHub raw URLs — the repository ships no data files.

---

## 8. Reproducibility

- ✅ Public data only — loaded from raw GitHub URLs
- ✅ No API keys, tokens, or manual uploads
- ✅ Fixed `random_state` for deterministic splits and models
- ✅ Runs end-to-end with `Runtime → Run all`

---

## 9. Results

*Populate this section after running the notebook.*

| Model | Precision (late) | Recall (late) | F1 (late) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| Gradient Boosting | — | — | — | — |

**Key findings:** _summarize the most predictive features and the best-performing model here._

---

## 10. Next Steps

- Hyperparameter tuning (grid / randomized search, cross-validation)
- Class-imbalance handling (class weights, SMOTE, threshold tuning)
- Feature importance / SHAP analysis for interpretability
- Geolocation-based distance features (customer ↔ seller)
- Deployment as a lightweight prediction service

---

## 11. License & Attribution

Dataset: **Brazilian E-Commerce Public Dataset by Olist**, released publicly on Kaggle.
Academic project developed for **PUC-Rio — Postgraduate Program in Machine Learning and Data Science**.

**Author:** Denilson Marcos · `4052025001353`
