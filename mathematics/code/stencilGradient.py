"""Gradient and Laplacian of an array via finite-difference stencils."""
import numpy as np


def gradient_1d(f, h=1.0):
    """First derivative: central in the interior, one-sided at the edges."""
    g = np.empty_like(f, dtype=float)
    g[1:-1] = (f[2:] - f[:-2]) / (2 * h)   # central difference
    g[0]    = (f[1] - f[0]) / h            # forward difference
    g[-1]   = (f[-1] - f[-2]) / h          # backward difference
    return g


def laplacian_2d(f, h=1.0):
    """Five-point stencil on the interior of a 2D grid."""
    lap = np.zeros_like(f, dtype=float)
    lap[1:-1, 1:-1] = (
        f[2:, 1:-1] + f[:-2, 1:-1] +
        f[1:-1, 2:] + f[1:-1, :-2] -
        4 * f[1:-1, 1:-1]
    ) / h ** 2
    return lap


if __name__ == "__main__":
    x = np.linspace(0, np.pi, 100)
    h = x[1] - x[0]
    # d/dx sin(x) = cos(x); compare the stencil against the analytic result
    err = np.abs(gradient_1d(np.sin(x), h) - np.cos(x)).max()
    print(f"max gradient error: {err:.2e}")  # O(h) at the edges
