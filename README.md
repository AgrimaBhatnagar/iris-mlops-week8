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



