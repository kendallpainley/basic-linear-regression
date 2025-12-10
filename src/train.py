
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import LinearRegressionScratch

def generate_synthetic(n=200, noise=0.5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, size=(n, 1))
    true_w0, true_w1 = 2.0, 3.5
    y = true_w0 + true_w1 * X[:, 0] + rng.normal(0, noise, size=n)
    return X, y

def load_csv(path: str, x_col: str, y_col: str):
    df = pd.read_csv(path)
    X = df[[x_col]].values
    y = df[y_col].values
    return X, y

def main(args):
    if args.csv and os.path.exists(args.csv):
        X, y = load_csv(args.csv, args.x_col, args.y_col)
        dataset_name = os.path.basename(args.csv)
    else:
        X, y = generate_synthetic(n=args.n_samples, noise=args.noise, seed=args.seed)
        dataset_name = f"synthetic_n{args.n_samples}_noise{args.noise}"

    model = LinearRegressionScratch(
        fit_intercept=True,
        method=args.method,
        learning_rate=args.learning_rate,
        n_iters=args.n_iters,
    )
    model.fit(X, y)
    r2 = model.score_r2(X, y)

    print(f"R^2 on training data: {r2:.4f}")
    print(f"Parameters (theta): {model.theta_.ravel()}")

    plt.figure(figsize=(6,4))
    plt.scatter(X[:,0], y, s=16, alpha=0.7, label="data")
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color="tomato", label="fit")
    plt.title(f"Linear Regression — {dataset_name}\nR^2={r2:.3f}")
    plt.xlabel(args.x_col if args.csv else "X")
    plt.ylabel(args.y_col if args.csv else "y")
    plt.legend()
    out_path = os.path.join("images", f"fit_{dataset_name}.png")
    os.makedirs("images", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a basic linear regression model")
    parser.add_argument("--method", choices=["normal", "gd"], default="normal",
                        help="Fitting method: normal equation or gradient descent")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for gradient descent")
    parser.add_argument("--n_iters", type=int, default=1000,
                        help="Iterations for gradient descent")
    parser.add_argument("--csv", type=str, default="",
                        help="Path to CSV file (optional). If not provided, uses synthetic data")
    parser.add_argument("--x_col", type=str, default="x",
                        help="Name of feature column in CSV")
    parser.add_argument("--y_col", type=str, default="y",
                        help="Name of target column in CSV")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of synthetic samples")
    parser.add_argument("--noise", type=float, default=0.5,
                        help="Noise level for synthetic data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    main(args)

