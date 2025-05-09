import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class BEKK:
    def __init__(self, returns, asset_names=None, dates=None, standardize=False):
        self.original_returns = returns
        if standardize:
            self.returns = ((returns - returns.mean(axis=0)) / returns.std(axis=0)).to_numpy()
        else:
            self.returns = np.asarray(returns)

        self.T, self.k = self.returns.shape
        self.asset_names = asset_names if asset_names else [f"Asset {i+1}" for i in range(self.k)]
        self.dates = dates if dates is not None else np.arange(self.T)

    def init_params(self):
        C = np.eye(self.k)
        a = 0.1 * np.ones(self.k)
        g = 0.8 * np.ones(self.k)
        return np.hstack([C[np.tril_indices(self.k)], a, g])

    def transform_params(self, params):
        n_c = int(self.k * (self.k + 1) / 2)
        c = params[:n_c]
        a = params[n_c:n_c + self.k]
        g = params[n_c + self.k:]

        C = np.zeros((self.k, self.k))
        C[np.tril_indices(self.k)] = c
        A = np.diag(a)
        G = np.diag(g)
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

    def fit(self, maxiter=300):
        init = self.init_params()
        opt = minimize(self.log_likelihood, init, method='L-BFGS-B', options={'maxiter': maxiter})
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

    def plot_conditional_covariances(self, smooth_window=20):
        covariances = self.get_conditional_covariances()
        var_1 = np.array([H[0, 0] for H in covariances])
        var_2 = np.array([H[1, 1] for H in covariances])
        cov_12 = np.array([H[0, 1] for H in covariances])
        corr_12 = cov_12 / np.sqrt(var_1 * var_2)

        # Lissage avec moyenne glissante
        def smooth(x, window):
            return np.convolve(x, np.ones(window) / window, mode='same')

        var_1_smooth = smooth(var_1, smooth_window)
        var_2_smooth = smooth(var_2, smooth_window)
        cov_12_smooth = smooth(cov_12, smooth_window)
        corr_12_smooth = smooth(corr_12, smooth_window)

        fig, axs = plt.subplots(2, 2, figsize=(14, 8))

        axs[0, 0].plot(self.dates, var_1_smooth, color='red')
        axs[0, 0].set_title(f"Variability of({self.asset_names[0]})")

        axs[0, 1].plot(self.dates, corr_12_smooth, color='red')
        axs[0, 1].axhline(0, color='black', linestyle='dotted')
        axs[0, 1].set_title(f"Corr({self.asset_names[0]}, {self.asset_names[1]})")

        axs[1, 0].plot(self.dates, cov_12_smooth, color='red')
        axs[1, 0].axhline(0, color='black', linestyle='dotted')
        axs[1, 0].set_title(f"Cov({self.asset_names[1]}, {self.asset_names[0]})")

        axs[1, 1].plot(self.dates, var_2_smooth, color='red')
        axs[1, 1].set_title(f"Variablility of({self.asset_names[1]})")

        for ax in axs.flatten():
            ax.grid(True)
            if hasattr(self.dates[0], 'strftime'):
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
