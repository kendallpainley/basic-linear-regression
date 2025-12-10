
from dataclasses import dataclass
import numpy as np

def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])

@dataclass
class LinearRegressionScratch:
    fit_intercept: bool = True
    method: str = "normal"  # "normal" or "gd"
    learning_rate: float = 0.01
    n_iters: int = 1000
    theta_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        X_b = add_bias(X) if self.fit_intercept else X

        if self.method == "normal":
            XtX = X_b.T @ X_b
            lam = 1e-8
            I = np.eye(XtX.shape[0])
            self.theta_ = np.linalg.inv(XtX + lam * I) @ X_b.T @ y
        elif self.method == "gd":
            m, n = X_b.shape
            self.theta_ = np.zeros((n, 1))
            for _ in range(self.n_iters):
                preds = X_b @ self.theta_
                gradients = (2/m) * (X_b.T @ (preds - y))
                self.theta_ -= self.learning_rate * gradients
        else:
            raise ValueError("Unknown method. Use 'normal' or 'gd'.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        X = np.asarray(X)
        X_b = add_bias(X) if self.fit_intercept else X
        return (X_b @ self.theta_).ravel()

    def score_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

