from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor

class CompatibleCatBoost(RegressorMixin, BaseEstimator):
    """Scikit-learn-compatible adapter for CatBoost across sklearn releases."""
    def __init__(self, iterations=300, learning_rate=0.05, depth=6, random_seed=42):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed

    def fit(self, X, y):
        self.estimator_ = CatBoostRegressor(
            iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth, random_seed=self.random_seed, verbose=False
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self)
        return self.estimator_.predict(X)
