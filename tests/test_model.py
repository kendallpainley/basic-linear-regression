
import numpy as np
from src.model import LinearRegressionScratch

def test_normal_equation_fit():
    X = np.array([[0],[1],[2],[3],[4]], dtype=float)
    y = 2 + 3*X.ravel()
    model = LinearRegressionScratch(method="normal")
    model.fit(X, y)
    r2 = model.score_r2(X, y)
    assert r2 > 0.999
    assert np.allclose(model.theta_.ravel(), [2,3], atol=1e-6)

def test_gradient_descent_fit():
    X = np.array([[0],[1],[2],[3],[4]], dtype=float)
    y = 2 + 3*X.ravel()
    model = LinearRegressionScratch(method="gd", learning_rate=0.05, n_iters=5000)
    model.fit(X, y)
    r2 = model.score_r2(X, y)
    assert r2 > 0.999

