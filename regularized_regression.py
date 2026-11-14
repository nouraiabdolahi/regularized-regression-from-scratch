import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

class RegularizedRegression:
    def __init__(self, alpha=1.0, method='ridge', max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = 0.0

    def _soft_threshold(self, rho, lam):
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0.0

    def fit(self, X, y):
        n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.mean(y)
        
        y_centered = y - self.bias
        
        if self.method == 'ridge':
            identity = np.eye(n_features)
            self.weights = np.linalg.inv(X.T @ X + self.alpha * identity) @ (X.T @ y_centered)
            
        elif self.method == 'lasso':
            col_norms = np.sum(X**2, axis=0)
            
            for _ in range(self.max_iter):
                weights_old = self.weights.copy()
                
                for j in range(n_features):
                    residual = y_centered - (X @ self.weights - X[:, j] * self.weights[j])
                    rho = np.dot(X[:, j], residual)
                    
                    if col_norms[j] == 0:
                        self.weights[j] = 0.0
                    else:
                        self.weights[j] = self._soft_threshold(rho, self.alpha) / col_norms[j]
                
                if np.max(np.abs(self.weights - weights_old)) < self.tol:
                    break
        else:
            raise ValueError("Method must be 'ridge' or 'lasso'")

    def predict(self, X):
        return X @ self.weights + self.bias


if __name__ == "__main__":
    data = fetch_california_housing()
    X = data.data[:500]
    y = data.target[:500]
    feature_names = data.feature_names
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std

    alphas = np.logspace(-2, 2, 40)
    lasso_weights = []
    
    for a in alphas:
        model = RegularizedRegression(alpha=a, method='lasso')
        model.fit(X_scaled, y)
        lasso_weights.append(model.weights)
        
    lasso_weights = np.array(lasso_weights)

    plt.figure(figsize=(9, 5))
    for i in range(X.shape[1]):
        plt.plot(alphas, lasso_weights[:, i], label=feature_names[i])
        
    plt.xscale('log')
    plt.xlabel('Regularization Strength (Alpha)')
    plt.ylabel('Coefficients')
    plt.title('Lasso Regularization Path (Sparsity Analysis)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig('lasso_path.png', dpi=300)
    plt.show()

    test_alpha = 10.0
    ridge = RegularizedRegression(alpha=test_alpha, method='ridge')
    ridge.fit(X_scaled, y)

    lasso = RegularizedRegression(alpha=test_alpha, method='lasso')
    lasso.fit(X_scaled, y)

    print("\nCoefficients Comparison at alpha = 10.0:")
    print(f"{'Feature':<15} | {'Ridge':<10} | {'Lasso':<10}")
    print("-" * 40)
    for name, w_r, w_l in zip(feature_names, ridge.weights, lasso.weights):
        print(f"{name:<15} | {w_r:<10.4f} | {w_l:<10.4f}")
