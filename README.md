
# Basic Linear Regression

A minimal, from-scratch implementation of linear regression in Python using NumPy, with two training options: the **Normal Equation** and **Gradient Descent**. Includes a training script that can generate synthetic data or load a CSV, produce an R² score, and save a plot of the fitted line.

## Features
- Linear regression implemented from origin
- Fitting method: normal equation or gradient descent
- Synthetic data generation or CSV loading
- R² metric, plots saved under `images/`
- Basic unit tests

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
python src/train.py --method normal --n_samples 300 --noise 0.7
python src/train.py --method gd --learning_rate 0.05 --n_iters 5000
python src/train.py --csv data/your_data.csv --x_col x --y_col y

