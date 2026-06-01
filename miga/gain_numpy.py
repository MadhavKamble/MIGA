"""
GAIN: Generative Adversarial Imputation Networks — pure NumPy implementation.
Yoon, Jordon & van der Schaar (2018), ICML.

No PyTorch/TensorFlow dependency. Uses manual backpropagation through
3-layer MLPs with ReLU + Sigmoid activations and Adam optimiser.

Interface:  GAIN_NP().fit_transform(X_with_nan) -> X_complete
"""

import numpy as np


# ─── Activations ────────────────────────────────────────────────────────────

def _relu(x):
    return np.maximum(0.0, x)

def _relu_grad(x):
    return (x > 0).astype(float)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ─── MLP ─────────────────────────────────────────────────────────────────────

class _MLP:
    """3-layer MLP with ReLU hidden + Sigmoid output.  Adam optimizer."""

    def __init__(self, in_dim: int, h_dim: int, out_dim: int, rng):
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / h_dim)
        self.W1 = rng.standard_normal((in_dim, h_dim)) * scale1
        self.b1 = np.zeros(h_dim)
        self.W2 = rng.standard_normal((h_dim, h_dim)) * scale2
        self.b2 = np.zeros(h_dim)
        self.W3 = rng.standard_normal((h_dim, out_dim)) * scale2
        self.b3 = np.zeros(out_dim)
        # Adam moments
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return {"W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2,
                "W3": self.W3, "b3": self.b3}

    def forward(self, x: np.ndarray) -> tuple:
        """Returns (output, cache) where cache stores intermediates for backprop."""
        z1 = x @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _relu(z2)
        z3 = a2 @ self.W3 + self.b3
        out = _sigmoid(z3)
        return out, (x, z1, a1, z2, a2, z3, out)

    def backward(self, d_out: np.ndarray, cache: tuple) -> dict:
        """Compute gradients given upstream gradient d_out."""
        x, z1, a1, z2, a2, z3, out = cache
        n = x.shape[0]

        # Layer 3
        d_z3 = d_out * out * (1 - out)          # sigmoid grad
        d_W3 = (a2.T @ d_z3) / n
        d_b3 = d_z3.mean(axis=0)
        d_a2 = d_z3 @ self.W3.T

        # Layer 2
        d_z2 = d_a2 * _relu_grad(z2)
        d_W2 = (a1.T @ d_z2) / n
        d_b2 = d_z2.mean(axis=0)
        d_a1 = d_z2 @ self.W2.T

        # Layer 1
        d_z1 = d_a1 * _relu_grad(z1)
        d_W1 = (x.T @ d_z1) / n
        d_b1 = d_z1.mean(axis=0)
        d_x  = d_z1 @ self.W1.T

        return {"W1": d_W1, "b1": d_b1,
                "W2": d_W2, "b2": d_b2,
                "W3": d_W3, "b3": d_b3,
                "_d_x": d_x}

    def step(self, grads: dict, lr: float = 1e-3,
             beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """Adam update."""
        self._t += 1
        t = self._t
        for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            g = grads[key]
            self._m[key] = beta1 * self._m[key] + (1 - beta1) * g
            self._v[key] = beta2 * self._v[key] + (1 - beta2) * g ** 2
            m_hat = self._m[key] / (1 - beta1 ** t)
            v_hat = self._v[key] / (1 - beta2 ** t)
            p = getattr(self, key)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ─── GAIN ────────────────────────────────────────────────────────────────────

def _normalise(X):
    X_min = np.nanmin(X, axis=0)
    X_max = np.nanmax(X, axis=0)
    rng_  = X_max - X_min
    rng_[rng_ < 1e-10] = 1.0
    return (X - X_min) / rng_, X_min, rng_


def _denormalise(X_norm, X_min, rng_):
    return X_norm * rng_ + X_min


class GAIN_NP:
    """
    Pure NumPy GAIN imputer (no PyTorch required).

    Parameters
    ----------
    hint_rate  : fraction of mask bits revealed to D (default 0.9)
    alpha      : MSE reconstruction weight (default 100)
    h_dim      : hidden layer size (default 256)
    iterations : training iterations (default 10000)
    batch_size : mini-batch size (default 128)
    lr         : Adam learning rate (default 1e-3)
    seed       : random seed
    verbose    : print loss every 1000 iters
    """

    def __init__(
        self,
        hint_rate: float = 0.9,
        alpha: float = 100.0,
        h_dim: int = 256,
        iterations: int = 10000,
        batch_size: int = 128,
        lr: float = 1e-3,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.hint_rate  = hint_rate
        self.alpha      = alpha
        self.h_dim      = h_dim
        self.iterations = iterations
        self.batch_size = batch_size
        self.lr         = lr
        self.seed       = seed
        self.verbose    = verbose

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n, dim = X.shape

        # 1. Normalise
        X_norm, X_min, X_rng = _normalise(X)
        M = (~np.isnan(X)).astype(float)
        X_tilde = np.where(M.astype(bool), X_norm, 0.0)

        # 2. Build G and D
        G = _MLP(2 * dim, self.h_dim, dim, rng)
        D = _MLP(2 * dim, self.h_dim, dim, rng)

        bs = min(self.batch_size, n)

        for step in range(self.iterations):
            idx = rng.integers(0, n, size=bs)
            xb  = X_tilde[idx].astype(float)
            mb  = M[idx].astype(float)

            # Noise for missing positions
            zb = rng.uniform(0, 0.01, size=xb.shape)
            x_in = mb * xb + (1 - mb) * zb

            # Hint matrix
            H_raw = rng.binomial(1, self.hint_rate, size=mb.shape).astype(float)
            H  = mb * H_raw + 0.5 * (1 - H_raw)

            # ── Discriminator forward ──────────────────────────────────────
            G_inp = np.concatenate([x_in, mb], axis=1)
            G_out, _ = G.forward(G_inp)
            X_hat  = mb * xb + (1 - mb) * G_out

            D_inp  = np.concatenate([X_hat, H], axis=1)
            D_out, D_cache = D.forward(D_inp)

            eps_ = 1e-8
            # BCE on observed (should be 1) and missing (should be 0)
            d_loss_obs  = -mb * np.log(D_out + eps_)
            d_loss_miss = -(1 - mb) * np.log(1 - D_out + eps_)
            d_loss_val  = (d_loss_obs + d_loss_miss).mean()

            # Gradient of BCE: dL/d(D_out)
            d_D_out = (mb * (-1 / (D_out + eps_)) +
                       (1 - mb) * (1 / (1 - D_out + eps_))) / (bs * dim)
            d_D_grads = D.backward(d_D_out, D_cache)
            D.step(d_D_grads, lr=self.lr)

            # ── Generator forward ──────────────────────────────────────────
            G_out2, G_cache = G.forward(G_inp)
            X_hat2 = mb * xb + (1 - mb) * G_out2

            D_inp2  = np.concatenate([X_hat2, H], axis=1)
            D_out2, D_cache2 = D.forward(D_inp2)

            # G adversarial loss: fool D on missing positions
            g_adv_val = -(1 - mb) * np.log(D_out2 + eps_)
            # G MSE loss on observed positions
            g_mse_val = mb * (xb - G_out2) ** 2
            g_mse_mean = g_mse_val.mean() / (mb.mean() + eps_)

            # Gradient of G's adversarial term wrt D_out2
            d_D_out2_for_G = (1 - mb) * (-1 / (D_out2 + eps_)) / (bs * dim)
            d_X_hat2 = D.backward(d_D_out2_for_G, D_cache2)["_d_x"][:, :dim]

            # Only the missing part of X_hat2 came from G
            d_G_from_adv = d_X_hat2 * (1 - mb)

            # Gradient of MSE wrt G_out2 (on observed positions)
            d_G_from_mse = self.alpha * (-2 * mb * (xb - G_out2)) / (bs * dim * (mb.mean() + eps_))

            d_G_total = d_G_from_adv + d_G_from_mse
            G_grads = G.backward(d_G_total, G_cache)
            G.step(G_grads, lr=self.lr)

            if self.verbose and (step + 1) % 1000 == 0:
                print(f"  iter {step+1:5d}  D={d_loss_val:.4f}  "
                      f"G_adv={g_adv_val.mean():.4f}  G_mse={g_mse_mean:.4f}")

        # 3. Impute full dataset
        G_inp_full = np.concatenate([X_tilde, M], axis=1)
        G_out_full, _ = G.forward(G_inp_full)
        X_imp_norm = M * X_tilde + (1 - M) * G_out_full
        return _denormalise(X_imp_norm, X_min, X_rng)
