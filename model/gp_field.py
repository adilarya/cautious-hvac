"""Layer 2 -- Gaussian-process spatial field estimator (3D).

Given S sparse sensor readings (3D locations + temperatures), fits a GP
with squared-exponential kernel and returns a dense posterior mean and
standard deviation over every voxel of the room.

Posterior at query point x*:
    mu(x*)  = K(x*, X) [K(X, X) + sigma^2 I]^{-1} y
    var(x*) = k(x*, x*) - K(x*, X) [K(X, X) + sigma^2 I]^{-1} K(X, x*)

Kernel: ConstantKernel * RBF(length_scale) + WhiteKernel. Hyperparameters
are optimised at fit() time by maximising the log marginal likelihood
(sklearn does this automatically with n_restarts_optimizer).
"""
from __future__ import annotations
import warnings
from typing import Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


class GPField:
    """GP regression from sparse 3D sensors to a dense voxel field."""

    def __init__(self,
                 nx: int = 10, ny: int = 8, nz: int = 5,
                 length_scale: float = 2.0):
        self.nx, self.ny, self.nz = nx, ny, nz

        kernel = (
            ConstantKernel(1.0, (0.1, 50.0))
            * RBF(length_scale=length_scale, length_scale_bounds=(0.3, 15.0))
            + WhiteKernel(noise_level=0.2, noise_level_bounds=(1e-3, 2.0))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1,
            normalize_y=True,
            alpha=0.0,
        )

        # Dense query grid over all voxels: shape (nx*ny*nz, 3)
        xx, yy, zz = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
        self.grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(float)

        self._fitted = False

    def fit(self, sensor_locs: np.ndarray, sensor_temps: np.ndarray) -> "GPField":
        """Fit GP to current sensor observations.

        sensor_locs  : (S, 3) -- voxel coordinates of sensors
        sensor_temps : (S,)   -- noisy temperature readings
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(np.asarray(sensor_locs, dtype=float),
                        np.asarray(sensor_temps, dtype=float))
        self._fitted = True
        return self

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean_field, std_field) each shaped (nx, ny, nz)."""
        if not self._fitted:
            raise RuntimeError("GPField.predict() called before fit()")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self.gp.predict(self.grid_points, return_std=True)
        shape = (self.nx, self.ny, self.nz)
        return mean.reshape(shape), std.reshape(shape)
