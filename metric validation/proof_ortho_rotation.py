import numpy as np
import shesha
import shesha.sim as sim


if __name__ == "__main__":
    rng = np.random.default_rng(320)
    n, k, d = 200, 5, 64
    Z = rng.standard_normal((n, k))
    W = rng.standard_normal((k, d))
    X = Z @ W + 0.01 * rng.standard_normal((n, d))
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Q = Vt.T
    Y = X @ Q
    shesha.feature_split(X)
    shesha.feature_split(Y)
    cka = sim.cka(X, Y)
    print(f"CKA after orthogonal rotation: {cka:.6f}")