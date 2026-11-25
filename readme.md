# Classification Experiments & Takeaways

Hands-on classification workflows across tabular data, raw text, and image pixels. Rather than standard tutorials, these notebooks document practical lessons around data leakage, class imbalance, and metric trade-offs.

## Notebooks & Core Learnings

| Notebook | Focus | Key Takeaway |
|---|---|---|
| [MNIST](./mnist_num_classification.ipynb) | Digits (784 pixels) | Pixel-shift data augmentation improved generalization far more than switching classifiers. |
| [Employee Attrition (Benchmark)](./emp_attrition_multiple_models.ipynb) | HR turnover | 85% accuracy was useless on imbalanced data; recall for the minority class had to drive decisions. |
| [Employee Attrition (Deep Dive)](./emp_attrition_kaggle_gold.ipynb) | Leak-free workflow | Splitting before any EDA or transformation is essential; standard SMOTE breaks categorical columns, SMOTENC does not. |
| [Spam Detector](./spam_detector.ipynb) | Email classification | Custom pipeline transformers for headers and HTML stripped noise without leaking test data into training folds. |
| [Titanic Survival](./titanic_survivor_predictor.ipynb) | Mixed tabular data | Simple feature engineering (titles, family size) beat out marginal gains from tuning Random Forest hyperparameters. |

## What Stood Out

- **Accuracy obscures failure:** In skewed datasets (like attrition), default models achieve strong accuracy by simply ignoring the minority class. 
- **Pipelines prevent subtle leakage:** Imputing, scaling, or oversampling before a train/test split silently pollutes validation metrics. All transformations belong inside isolated folds.
- **Data preparation beats hyperparameter tuning:** Across text, tabular, and image data, meaningful feature engineering and augmentation yielded larger performance jumps than algorithm swapping.
- **Metric choice depends on error cost:** False positives in spam filtering cost real emails; false negatives in attrition cost retention opportunities. Optimize for the cost of the error, not the overall score.

## Repository Structure

```text
classification-problems/
├── mnist_num_classification.ipynb
├── emp_attrition_multiple_models.ipynb
├── emp_attrition_kaggle_gold.ipynb
├── spam_detector.ipynb
├── titanic_survivor_predictor.ipynb
└── data/
    ├── emp.csv
    ├── titanic/
    └── spamassassin/
```

*Note: MNIST downloads automatically via OpenML on first execution.*

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install jupyter numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook
```