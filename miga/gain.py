"""
GAIN: Generative Adversarial Imputation Networks
Yoon, Jordon & van der Schaar (2018), ICML.

PyTorch implementation following the original paper exactly:
  - Generator G(X̃, Z, M) → imputed values
  - Discriminator D(X̃, H) → per-cell probability of being real
  - Hint matrix H reveals mask bits with probability hint_rate

Key hyperparameters (paper defaults):
  hint_rate = 0.9    (fraction of mask bits revealed to D)
  alpha     = 100    (MSE loss weight in G's objective)
  iterations = 10000
  batch_size = 128

Interface mirrors sklearn: fit_transform(X_missing) → X_complete
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _normalise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalise to [0,1] using column stats of non-NaN values."""
    X_min = np.nanmin(X, axis=0)
    X_max = np.nanmax(X, axis=0)
    rng = X_max - X_min
    rng[rng < 1e-10] = 1.0
    return (X - X_min) / rng, X_min, rng


def _denormalise(X_norm: np.ndarray, X_min: np.ndarray, rng: np.ndarray) -> np.ndarray:
    return X_norm * rng + X_min


class _Generator(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        # Input: concatenated [X̃ (dim), M (dim)] = 2*dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim),  nn.ReLU(),
            nn.Linear(h_dim, dim),    nn.Sigmoid(),
        )

    def forward(self, x_tilde: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_tilde, m], dim=1)
        return self.net(inp)


class _Discriminator(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        # Input: concatenated [X̂ (dim), H (dim)] = 2*dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim),  nn.ReLU(),
            nn.Linear(h_dim, dim),    nn.Sigmoid(),
        )

    def forward(self, x_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_hat, h], dim=1)
        return self.net(inp)


class GAIN:
    """
    GAIN imputer.

    Parameters
    ----------
    hint_rate  : fraction of mask bits revealed to discriminator (default 0.9)
    alpha      : MSE reconstruction weight in generator loss (default 100)
    h_dim      : hidden layer size (default 256)
    iterations : training iterations (default 10000)
    batch_size : mini-batch size (default 128)
    seed       : random seed
    verbose    : print loss every 1000 iterations
    """

    def __init__(
        self,
        hint_rate: float = 0.9,
        alpha: float = 100.0,
        h_dim: int = 256,
        iterations: int = 10000,
        batch_size: int = 128,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.hint_rate  = hint_rate
        self.alpha      = alpha
        self.h_dim      = h_dim
        self.iterations = iterations
        self.batch_size = batch_size
        self.seed       = seed
        self.verbose    = verbose

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values (NaN) in X.

        Parameters
        ----------
        X : (n, p) array with np.nan marking missing values

        Returns
        -------
        X_imputed : (n, p) complete array
        """
        rng = np.random.default_rng(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)

        n, dim = X.shape

        # ── 1. Normalise to [0, 1] ──────────────────────────────────────────
        X_norm, X_min, X_rng = _normalise(X)

        # Mask: M[i,j]=1 if observed, 0 if missing
        M = (~np.isnan(X)).astype(np.float32)

        # X̃: observed values kept, missing filled with 0
        X_tilde = np.where(M.astype(bool), X_norm, 0.0).astype(np.float32)

        # ── 2. Build networks ───────────────────────────────────────────────
        G = _Generator(dim, self.h_dim)
        D = _Discriminator(dim, self.h_dim)

        opt_G = optim.Adam(G.parameters())
        opt_D = optim.Adam(D.parameters())

        bce = nn.BCELoss(reduction="mean")

        X_t = torch.from_numpy(X_tilde)
        M_t = torch.from_numpy(M)

        # ── 3. Training loop ────────────────────────────────────────────────
        for step in range(self.iterations):
            # Mini-batch
            idx = rng.integers(0, n, size=min(self.batch_size, n))
            xb = X_t[idx]
            mb = M_t[idx]

            # Noise injected into missing positions (paper Eq.1)
            zb = torch.from_numpy(rng.uniform(0, 0.01, size=xb.shape).astype(np.float32))
            x_in = mb * xb + (1 - mb) * zb

            # Hint matrix H: reveal mask with prob hint_rate, else 0.5
            H_raw = rng.binomial(1, self.hint_rate, size=mb.shape).astype(np.float32)
            H = torch.from_numpy(mb.numpy() * H_raw + 0.5 * (1 - H_raw))

            # ── Discriminator step ─────────────────────────────────────────
            G.eval(); D.train()
            with torch.no_grad():
                G_out = G(x_in, mb)
                X_hat = mb * xb + (1 - mb) * G_out   # keep observed, fill missing

            D_out = D(X_hat.detach(), H)
            # D trained only on missing positions (1-M)
            d_loss = bce(D_out * (1 - mb), (1 - mb) * torch.zeros_like(mb)) + \
                     bce(D_out * mb,        mb * torch.ones_like(mb))

            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # ── Generator step ─────────────────────────────────────────────
            G.train(); D.eval()
            G_out = G(x_in, mb)
            X_hat = mb * xb + (1 - mb) * G_out

            D_out = D(X_hat, H)
            # Adversarial: fool D on missing positions
            g_adv  = bce(D_out * (1 - mb), (1 - mb) * torch.ones_like(mb))
            # MSE reconstruction on *observed* positions only
            g_mse  = torch.mean(mb * (xb - G_out) ** 2) / (torch.mean(mb) + 1e-8)
            g_loss = g_adv + self.alpha * g_mse

            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            if self.verbose and (step + 1) % 1000 == 0:
                print(f"  iter {step+1:5d}  D={d_loss.item():.4f}  G={g_loss.item():.4f}")

        # ── 4. Impute full dataset ──────────────────────────────────────────
        G.eval()
        with torch.no_grad():
            G_full = G(X_t, M_t).numpy()

        X_imputed_norm = M * X_tilde + (1 - M) * G_full
        return _denormalise(X_imputed_norm, X_min, X_rng)
