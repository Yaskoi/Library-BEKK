import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class BEKK:
    def __init__(self, returns, asset_names=None):
        self.returns = np.asarray(returns)
        self.T, self.k = self.returns.shape
        self.asset_names = asset_names if asset_names else [f"Asset {i+1}" for i in range(self.k)]

    def vech(self, M):
        return M[np.tril_indices(self.k)]

    def unvec(self, v):
        M = np.zeros((self.k, self.k))
        tril = np.tril_indices(self.k)
        M[tril] = v
        return M

    def init_params(self):
        C = np.eye(self.k)
        A = 0.1 * np.eye(self.k)
        G = 0.8 * np.eye(self.k)
        return np.hstack([self.vech(C), A.flatten(), G.flatten()])

    def transform_params(self, params):
        n_c = int(self.k * (self.k + 1) / 2)
        c = params[:n_c]
        a = params[n_c:n_c + self.k**2]
        g = params[n_c + self.k**2:]

        C = self.unvec(c)
        A = a.reshape(self.k, self.k)
        G = g.reshape(self.k, self.k)
        return C, A, G

    def log_likelihood(self, params):
        C, A, G = self.transform_params(params)
        H_t = np.cov(self.returns.T)
        logL = 0

        for t in range(self.T):
            r_t = self.returns[t].reshape(-1, 1)
            H_t = C @ C.T + A @ r_t @ r_t.T @ A.T + G @ H_t @ G.T
            try:
                inv_H = np.linalg.inv(H_t)
                sign, logdet = np.linalg.slogdet(H_t)
                if sign <= 0:
                    return 1e10
                logL += 0.5 * (np.log(2 * np.pi) * self.k + logdet + (r_t.T @ inv_H @ r_t))
            except np.linalg.LinAlgError:
                return 1e10

        return logL.item()

    def fit(self):
        init = self.init_params()
        opt = minimize(self.log_likelihood, init, method='L-BFGS-B')
        self.params = opt.x
        self.C, self.A, self.G = self.transform_params(self.params)
        return opt

    def get_conditional_covariances(self):
        H_t = np.cov(self.returns.T)
        covariances = []
        for t in range(self.T):
            r_t = self.returns[t].reshape(-1, 1)
            H_t = self.C @ self.C.T + self.A @ r_t @ r_t.T @ self.A.T + self.G @ H_t @ self.G.T
            covariances.append(H_t)
        return np.array(covariances)

    def plot_conditional_covariances(self):
        covariances = self.get_conditional_covariances()
        fig, axs = plt.subplots(self.k, self.k, figsize=(12, 6))

        for i in range(self.k):
            for j in range(self.k):
                if self.k == 2:
                    ax = axs[i, j]
                else:
                    ax = axs[i][j]
                ax.plot([H[i, j] for H in covariances])
                if i == j:
                    ax.set_title(f"Variance conditionnelle: {self.asset_names[i]}")
                else:
                    ax.set_title(f"Covariance: {self.asset_names[i]}-{self.asset_names[j]}")
        plt.tight_layout()
        plt.show()