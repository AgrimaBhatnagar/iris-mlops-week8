# IRIS Data Poisoning Experiment with MLflow

This project performs **data poisoning attacks** on the classic Iris dataset at different severity levels (5%, 10%, 50%), trains a machine learning model on the corrupted data, and evaluates the degradation in performance using **MLflow experiment tracking**.

The assignment demonstrates:

- How to introduce random noise–based poisoning into a dataset.
- How model accuracy and class-level predictions deteriorate as poisoning increases.
- How to log parameters, metrics, artifacts, and models using MLflow.
- Best practices to mitigate data poisoning risks.
- How data quantity requirements increase when data quality decreases.

---

## 📌 1. Project Structure
iris-poisoning/
│── train.py # Main training + poisoning script
│── run_experiments.sh # Runs poisoning levels 0%, 5%, 10%, 50%
│── requirements.txt # Python dependencies
│── README.md # You are reading this file
│── .gitignore
└── mlruns/ # MLflow experiment storage (auto-created)

## 📌 2. Setup Instructions

### Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
## 📌 3. Running MLflow UI
mlflow ui --port 5000

## 📌 4. Running the Experiments
./run_experiments.sh
📌 5. How Data Poisoning Works

A percentage of the training samples is selected and their feature values are replaced with random values within real feature ranges.

Poison levels:

0% — Clean dataset

5% — Light corruption

10% — Moderate corruption

50% — Heavy corruption (half the dataset becomes noise)

The validation set remains clean to correctly evaluate model degradation.

📌 6. Observed Outcomes
✔ 0% Poisoning

Accuracy ~95%

Clean decision boundaries

✔ 5% Poisoning

Slight accuracy drop (2–4%)

Some confusion between Versicolor & Virginica

✔ 10% Poisoning

Noticeable degradation

Confusion matrix becomes noisier

F1 score drops significantly

✔ 50% Poisoning

Model accuracy collapses to ~33% (random guess for 3 classes)

Predictions become unstable

MLflow artifacts clearly show failure patterns

📌 7. Mitigation Strategies Against Data Poisoning
1. Data Validation Pipelines

Detect impossible values

Check feature distributions

Use schema validation (e.g., Great Expectations)

2. Outlier Detection

Isolation Forest

One-Class SVM

Local Outlier Factor (LOF)

3. Data Provenance Tracking

Log data source metadata

Track dataset versioning using DVC

Maintain audit trails

4. Model-Level Defenses

Robust loss functions (Huber, Tukey)

Label smoothing

Differential privacy training

5. Increase Clean Training Data

When a dataset is partially poisoned, more clean data is needed to weaken the attack’s effect.

📌 8. Data Quantity vs Data Quality

When data quality decreases due to poisoning, accuracy can be preserved only if more clean data is added.

Example:

Poison Level	Extra Clean Data Needed	Reason
5%	~10% more	Mild corruption
10%	30–50% more	Boundaries start collapsing
50%	3–5× more	Noise overwhelms signal

Key Insight:

With low data quality, you need higher data quantity to maintain accuracy.



