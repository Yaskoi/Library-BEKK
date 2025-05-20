import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class DiagonalBEKK:
    def __init__(self, returns, p=1, q=1, asset_names=None, dates=None, standardize=False):
        self.original_returns = returns
        if standardize:
            self.returns = ((returns - returns.mean(axis=0)) / returns.std(axis=0)).to_numpy()
        else:
            self.returns = np.asarray(returns)

        self.T, self.k = self.returns.shape
        self.p = p  # ARCH lags
        self.q = q  # GARCH lags
        self.asset_names = asset_names if asset_names else [f"Asset {i+1}" for i in range(self.k)]
        self.dates = dates if dates is not None else np.arange(self.T)

    def init_params(self):
        C = np.eye(self.k)
        a = 0.1 * np.ones((self.p, self.k))
        g = 0.8 * np.ones((self.q, self.k))
        return np.hstack([C[np.tril_indices(self.k)], a.flatten(), g.flatten()])

    def transform_params(self, params):
        n_c = int(self.k * (self.k + 1) / 2)
        c = params[:n_c]
        a = params[n_c:n_c + self.k * self.p]
        g = params[n_c + self.k * self.p:]

        C = np.zeros((self.k, self.k))
        C[np.tril_indices(self.k)] = c
        A = np.array([np.diag(a[i*self.k:(i+1)*self.k]) for i in range(self.p)])
        G = np.array([np.diag(g[i*self.k:(i+1)*self.k]) for i in range(self.q)])
        return C, A, G

    def log_likelihood(self, params):
        C, A_list, G_list = self.transform_params(params)
        H_t = [np.cov(self.returns.T)] * max(1, self.q)
        eps_t = [self.returns[0].reshape(-1, 1)] * max(1, self.p)
        logL = 0

        for t in range(max(self.p, self.q), self.T):
            H = C @ C.T

            for i in range(self.p):
                r_lag = self.returns[t - i - 1].reshape(-1, 1)
                H += A_list[i] @ r_lag @ r_lag.T @ A_list[i].T

            for j in range(self.q):
                H += G_list[j] @ H_t[-j-1] @ G_list[j].T

            r_t = self.returns[t].reshape(-1, 1)
            try:
                inv_H = np.linalg.inv(H)
                sign, logdet = np.linalg.slogdet(H)
                if sign <= 0:
                    return 1e10
                logL += 0.5 * (np.log(2 * np.pi) * self.k + logdet + (r_t.T @ inv_H @ r_t))
            except np.linalg.LinAlgError:
                return 1e10

            H_t.append(H)
            if len(H_t) > self.q:
                H_t.pop(0)

        return logL.item()

    def fit(self, maxiter=300):
        init = self.init_params()
        opt = minimize(self.log_likelihood, init, method='L-BFGS-B', options={'maxiter': maxiter})
        self.params = opt.x
        self.C, self.A_list, self.G_list = self.transform_params(self.params)
        return opt

    def get_conditional_covariances(self):
        C, A_list, G_list = self.C, self.A_list, self.G_list
        H_t = [np.cov(self.returns.T)] * max(1, self.q)
        covariances = []

        for t in range(max(self.p, self.q), self.T):
            H = C @ C.T

            for i in range(self.p):
                r_lag = self.returns[t - i - 1].reshape(-1, 1)
                H += A_list[i] @ r_lag @ r_lag.T @ A_list[i].T

            for j in range(self.q):
                H += G_list[j] @ H_t[-j-1] @ G_list[j].T

            covariances.append(H)
            H_t.append(H)
            if len(H_t) > self.q:
                H_t.pop(0)

        return np.array(covariances)

    def plot_conditional_covariances(self, smooth_window=20):
        covariances = self.get_conditional_covariances()
        var_1 = np.array([H[0, 0] for H in covariances])
        var_2 = np.array([H[1, 1] for H in covariances])
        cov_12 = np.array([H[0, 1] for H in covariances])
        corr_12 = cov_12 / np.sqrt(var_1 * var_2)

        def smooth(x, window):
            return np.convolve(x, np.ones(window) / window, mode='same')

        var_1_smooth = smooth(var_1, smooth_window)
        var_2_smooth = smooth(var_2, smooth_window)
        cov_12_smooth = smooth(cov_12, smooth_window)
        corr_12_smooth = smooth(corr_12, smooth_window)

        fig, axs = plt.subplots(2, 2, figsize=(14, 8))

        axs[0, 0].plot(self.dates[max(self.p, self.q):], var_1_smooth, color='red')
        axs[0, 0].set_title(f"Variability of({self.asset_names[0]})")

        axs[0, 1].plot(self.dates[max(self.p, self.q):], corr_12_smooth, color='red')
        axs[0, 1].axhline(0, color='black', linestyle='dotted')
        axs[0, 1].set_title(f"Corr({self.asset_names[0]}, {self.asset_names[1]})")

        axs[1, 0].plot(self.dates[max(self.p, self.q):], cov_12_smooth, color='red')
        axs[1, 0].axhline(0, color='black', linestyle='dotted')
        axs[1, 0].set_title(f"Cov({self.asset_names[1]}, {self.asset_names[0]})")

        axs[1, 1].plot(self.dates[max(self.p, self.q):], var_2_smooth, color='red')
        axs[1, 1].set_title(f"Variability of({self.asset_names[1]})")

        for ax in axs.flatten():
            ax.grid(True)
            if hasattr(self.dates[0], 'strftime'):
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


class FullBEKK:
    def __init__(self, returns, p=1, q=1, asset_names=None, dates=None, standardize=False):
        self.original_returns = returns
        if standardize:
            self.returns = ((returns - returns.mean(axis=0)) / returns.std(axis=0)).to_numpy()
        else:
            self.returns = np.asarray(returns)

        self.T, self.k = self.returns.shape
        self.p = p  # ARCH lags
        self.q = q  # GARCH lags
        self.asset_names = asset_names if asset_names else [f"Asset {i+1}" for i in range(self.k)]
        self.dates = dates if dates is not None else np.arange(self.T)

    def vech(self, M):
        return M[np.tril_indices(self.k)]

    def unvec(self, v):
        M = np.zeros((self.k, self.k))
        tril = np.tril_indices(self.k)
        M[tril] = v
        return M

    def init_params(self):
        C = np.eye(self.k)
        A = [0.1 * np.eye(self.k) for _ in range(self.p)]
        G = [0.8 * np.eye(self.k) for _ in range(self.q)]
        return np.hstack([
            self.vech(C),
            *[a.flatten() for a in A],
            *[g.flatten() for g in G]
        ])

    def transform_params(self, params):
        n_c = int(self.k * (self.k + 1) / 2)
        c = params[:n_c]
        n_a = self.p * self.k**2
        n_g = self.q * self.k**2

        a = params[n_c:n_c + n_a]
        g = params[n_c + n_a:]

        C = self.unvec(c)
        A = [a[i*self.k**2:(i+1)*self.k**2].reshape(self.k, self.k) for i in range(self.p)]
        G = [g[i*self.k**2:(i+1)*self.k**2].reshape(self.k, self.k) for i in range(self.q)]

        return C, A, G

    def log_likelihood(self, params):
        C, A_list, G_list = self.transform_params(params)
        H_t = [np.cov(self.returns.T)] * max(1, self.q)
        logL = 0

        for t in range(max(self.p, self.q), self.T):
            H = C @ C.T

            for i in range(self.p):
                r_lag = self.returns[t - i - 1].reshape(-1, 1)
                H += A_list[i] @ r_lag @ r_lag.T @ A_list[i].T

            for j in range(self.q):
                H += G_list[j] @ H_t[-j-1] @ G_list[j].T

            r_t = self.returns[t].reshape(-1, 1)
            try:
                inv_H = np.linalg.inv(H)
                sign, logdet = np.linalg.slogdet(H)
                if sign <= 0:
                    return 1e10
                logL += 0.5 * (np.log(2 * np.pi) * self.k + logdet + (r_t.T @ inv_H @ r_t))
            except np.linalg.LinAlgError:
                return 1e10

            H_t.append(H)
            if len(H_t) > self.q:
                H_t.pop(0)

        return logL.item()

    def fit(self, maxiter=500):
        init = self.init_params()
        opt = minimize(self.log_likelihood, init, method='L-BFGS-B', options={'maxiter': maxiter})
        self.params = opt.x
        self.C, self.A_list, self.G_list = self.transform_params(self.params)
        return opt

    def get_conditional_covariances(self):
        C, A_list, G_list = self.C, self.A_list, self.G_list
        H_t = [np.cov(self.returns.T)] * max(1, self.q)
        covariances = []

        for t in range(max(self.p, self.q), self.T):
            H = C @ C.T

            for i in range(self.p):
                r_lag = self.returns[t - i - 1].reshape(-1, 1)
                H += A_list[i] @ r_lag @ r_lag.T @ A_list[i].T

            for j in range(self.q):
                H += G_list[j] @ H_t[-j-1] @ G_list[j].T

            covariances.append(H)
            H_t.append(H)
            if len(H_t) > self.q:
                H_t.pop(0)

        return np.array(covariances)

    def plot_conditional_covariances(self, smooth_window=20):
        covariances = self.get_conditional_covariances()
        var_1 = np.array([H[0, 0] for H in covariances])
        var_2 = np.array([H[1, 1] for H in covariances])
        cov_12 = np.array([H[0, 1] for H in covariances])
        corr_12 = cov_12 / np.sqrt(var_1 * var_2)

        def smooth(x, window):
            return np.convolve(x, np.ones(window) / window, mode='same')

        var_1_smooth = smooth(var_1, smooth_window)
        var_2_smooth = smooth(var_2, smooth_window)
        cov_12_smooth = smooth(cov_12, smooth_window)
        corr_12_smooth = smooth(corr_12, smooth_window)

        fig, axs = plt.subplots(2, 2, figsize=(14, 8))

        axs[0, 0].plot(self.dates[max(self.p, self.q):], var_1_smooth, color='red')
        axs[0, 0].set_title(f"Variability of({self.asset_names[0]})")

        axs[0, 1].plot(self.dates[max(self.p, self.q):], corr_12_smooth, color='red')
        axs[0, 1].axhline(0, color='black', linestyle='dotted')
        axs[0, 1].set_title(f"Corr({self.asset_names[0]}, {self.asset_names[1]})")

        axs[1, 0].plot(self.dates[max(self.p, self.q):], cov_12_smooth, color='red')
        axs[1, 0].axhline(0, color='black', linestyle='dotted')
        axs[1, 0].set_title(f"Cov({self.asset_names[1]}, {self.asset_names[0]})")

        axs[1, 1].plot(self.dates[max(self.p, self.q):], var_2_smooth, color='red')
        axs[1, 1].set_title(f"Variability of({self.asset_names[1]})")

        for ax in axs.flatten():
            ax.grid(True)
            if hasattr(self.dates[0], 'strftime'):
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
